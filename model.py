# models/model.py
import torch
import torch.nn as nn
import os
from typing import Union
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel,AutoModelForCausalLM 
from peft import get_peft_model, LoraConfig, TaskType


class CodebookAttention(nn.Module):
    def __init__(self, codebook_path, num_heads=2, dropout=0.1, freeze=False):
        """
        Args:
            codebook_path (str): Path to the pretrained codebook (.pt/.pth/.npy).
                                 预训练码本路径（.pt/.pth/.npy）
            num_heads (int): Number of attention heads.
                             attention 的头数
            dropout (float): Attention dropout.
                             attention dropout
            freeze (bool): Whether to freeze the codebook parameters (default: False).
                           是否冻结码本参数（默认 False）
        """
        super().__init__()

        # 加载嵌入矩阵
        if not os.path.exists(codebook_path):
            raise FileNotFoundError(f"Codebook file not found: {codebook_path}")

        if codebook_path.endswith(('.pt', '.pth')):
            embeddings = torch.load(codebook_path)
        elif codebook_path.endswith('.npy'):
            import numpy as np
            embeddings = torch.from_numpy(np.load(codebook_path))
        else:
            raise ValueError(f"Unsupported file format: {codebook_path}")

        if embeddings.dim() != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

        self.codebook_size, self.embedding_dim = embeddings.size()

        # 注册为 buffer 或 parameter,决定codebook是否更新，默认更新
        #  register codebook as buffer or parameter, decing whether update codebook via Back Propagation
        if freeze:
            self.register_buffer('codebook', embeddings)
        else:
            self.codebook = nn.Parameter(embeddings)

        # Cross attention layers (batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norm（适用于序列输入）
        self.norm = nn.LayerNorm(self.embedding_dim)

    def forward(self, z: torch.Tensor, z_effect_len: Union[torch.Tensor, None], mode: str, num_finite: int = 10):
        """
        Args:
            z (Tensor): Input tensor. Shape can be:
                        - (B, D) -> single vector
                        - (B, L, D) -> sequence input
                        输入张量，形状可以是：(B, D) 或 (B, L, D)。
            z_effect_len (Tensor): A 1D tensor of effective lengths for each sequence in z.
                                   一个记录z中每个序列有效长度的一维张量。
            mode (str): The operational mode. Options: 'original', 'soft_prompt', 'finite'. "original by default"
                        操作模式。选项：'original', 'soft_prompt', 'finite'。默认是original
            num (int): The number of groups for the 'finite' mode.
                       'finite' 模式下的分组数。

        Returns:
            output (Tensor): The output tensor with the same shape as the original input z.
                             输出张量，形状与原始输入 z 相同。
        """
        if mode not in ['original', 'soft_prompt', 'finite']:
            raise ValueError(f"Unsupported mode: {mode}. Must be one of 'original', 'soft_prompt', 'finite'.")

        original_shape = z.shape
        batch_size = z.shape[0]

        # 确保输入是三维张量，并处理z_effect_len
        # Ensure input is 3D and handle z_effect_len
        if z.dim() == 2:
            z = z.unsqueeze(1)
            if z_effect_len is None:
                z_effect_len = torch.ones(batch_size, dtype=torch.long, device=z.device)
        
        if z_effect_len is None:
            raise ValueError("z_effect_len must be provided for 3D input.")

        # 构造 key/value：每个 batch 都复制整个 codebook 
        # construct key/value
        key_value = self.codebook.expand(batch_size, -1, -1)  # (B, K, D)

        if mode == "original":
            # 模式 'original'：Query 是整个序列 z
            # Mode 'original': Query is the entire sequence z
            query = z
            output, _ = self.cross_attn(
                query=query,
                key=key_value,
                value=key_value
            )

        elif mode == "soft_prompt":
            # 模式 'soft_prompt'：Query 是每个有效序列的均值
            # Mode 'soft_prompt': Query is the mean of each effective sequence
            
            # 考虑到左侧填充，需要先提取有效部分再求均值
            # Account for left padding, extract effective part before calculating mean
            mean_z = []
            for i in range(batch_size):
                # 获取有效部分的起始索引
                # get the start idx of effective part
                start_idx = z.shape[1] - z_effect_len[i]
                effective_z = z[i, start_idx:, :]
                if effective_z.numel() > 0:
                    mean_z.append(effective_z.mean(dim=0))
                else:
                    # 对于空序列，使用一个零向量，以保持批次大小一致
                    # for empty sequence, use a zero-vector to make sure batch size shape
                    mean_z.append(torch.zeros(z.shape[2], device=z.device))
            
            query = torch.stack(mean_z).unsqueeze(1)  # (B, 1, D)
            
            output, _ = self.cross_attn(
                query=query,
                key=key_value,
                value=key_value
            )

        elif mode == "finite":
            # 模式 'finite'：将序列 z 分为 num 组，每组求均值作为 Query
            # Mode 'finite': Divide sequence z into 'num' groups, and use the mean of each group as a query
            
            grouped_queries = []
            for i in range(batch_size):
                effective_len = z_effect_len[i]
                start_idx = z.shape[1] - effective_len
                effective_z = z[i, start_idx:, :]
                
                group_queries = []
                if effective_len > 0:
                    # 计算每个组的长度
                    # calculate length of each group
                    group_size = torch.ceil(torch.tensor(effective_len / num_finite)).long()
                    for j in range(num_finite):
                        start_group_idx = j * group_size
                        end_group_idx = min((j + 1) * group_size, effective_len)
                        
                        if start_group_idx < end_group_idx:
                            group = effective_z[start_group_idx:end_group_idx]
                            group_queries.append(group.mean(dim=0))
                        else:
                            # 填充零向量
                            # pad zero vector
                            group_queries.append(torch.zeros(z.shape[2], device=z.device))
                else:
                    # 对于空序列，填充零向量
                    # pad zero vector
                    group_queries = [torch.zeros(z.shape[2], device=z.device)] * num_finite

                grouped_queries.append(torch.stack(group_queries))
            
            query = torch.stack(grouped_queries)  # (B, num, D)

            output, _ = self.cross_attn(
                query=query,
                key=key_value,
                value=key_value
            )

        attended_token = self.norm(output)
        
        # 将输出形状还原
        # Restore the output shape
        if mode == "original" and len(original_shape) == 2:
            return attended_token.squeeze(1)
        elif mode == "original" and len(original_shape) == 3:
            return attended_token
        elif mode == "soft_prompt":
            return attended_token.squeeze(1)
        elif mode == "finite":
            return attended_token

class AttentionPoolLayer(nn.Module):
    """
    改进的注意力池化层，支持特征维度不变
    An improved attention pooling layer that supports a constant feature dimension.
    
    功能说明：
    - 将每 pool_ratio 个连续 token 分组
    - 使用可学习的注意力机制对每组内的 token 进行加权聚合
    - 输出序列长度变为原来的 1/pool_ratio，但特征维度保持不变
    Function:
    - Group every pool_ratio consecutive tokens.

    - Apply a learnable attention mechanism to compute a weighted aggregation within each group.

    - The output sequence length becomes 1/pool_ratio of the original, while the feature dimension remains unchanged.
    """
    def __init__(self, hidden_dim=1024, pool_ratio=2):
        """
        初始化注意力池化层
        Initialize the attention pooling layer.

        参数：
        - hidden_dim: 输入特征的维度（如 BERT 的 768 或 1024）
        - pool_ratio: 每多少个 token 进行一次池化（例如 2 表示两两合并）
        Arguments:
        - hidden_dim: The dimensionality of the input features (e.g., 768 or 1024 for BERT).
        - pool_ratio: The number of tokens grouped together for each pooling step (e.g., 2 means merging every two tokens).
        """
        super().__init__()
        self.pool_ratio = pool_ratio  # 池化比例：每 pool_ratio 个 token 合并为 1 个
        # Pooling ratio: merge every `pool_ratio` tokens into one.

        # 线性层用于计算每个 token 的重要性得分（作为注意力权重）
        # Linear layer to compute importance score (as attention weight) for each token.
        self.query = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        前向传播函数
        Forward pass of the attention pooling layer.

        输入：
        - x: 形状为 [batch_size * seq_len, hidden_dim] 的张量
             注意：这里的 batch_size * seq_len 是展平后的序列长度
        Output:
        - pooled: 形状为 [batch_size * new_seq_len, hidden_dim] 的池化后张量
        Input:
        - x: A tensor of shape [batch_size * seq_len, hidden_dim].
        Note: batch_size * seq_len represents the flattened sequence length.
        Output:
        - pooled: A pooled tensor of shape [batch_size * new_seq_len, hidden_dim].
        """
        batch_size_total, hidden_dim = x.shape  # 获取总长度和特征维度
        # Total number of tokens and feature dimension.

        # 将输入 reshape 成 (N, pool_ratio, hidden_dim)，每组 pool_ratio 个 token
        # Reshape input into groups of `pool_ratio` tokens: [N, pool_ratio, hidden_dim]
        x = x.view(-1, self.pool_ratio, hidden_dim)
        # 示例：若原序列长为 10，pool_ratio=2，则变为 (5, 2, D)
        # Example: If the original sequence length is 10 and pool_ratio = 2, the shape becomes (5, 2, D).

        # 计算注意力权重：对每组中的 token 打分并归一化
        # Compute attention weights: score each token in the group and normalize with softmax.
        weights = torch.softmax(self.query(x), dim=1)  # shape: [N, pool_ratio, 1]
        # self.query(x) 输出形状为 [N, pool_ratio, 1]，softmax 沿着组内维度（dim=1）归一化
        # self.query(x) outputs a tensor of shape [N, pool_ratio, 1], and softmax is applied along the group dimension (dim=1) for normalization.
        

        # 加权求和：对每组中的 token 按注意力权重加权平均
        # Weighted sum: compute weighted average of tokens in each group.
        pooled = torch.sum(x * weights, dim=1)  # shape: [N, hidden_dim]
        # 输出序列长度变为原来的 1/pool_ratio，但维度不变
        # output sequence length changes to 1/pool_ratio of original length, but dimension unchange

        return pooled  # 返回池化后的表示
        # Return pooled representation with reduced sequence length.
    

class SSA_model(nn.Module):

    def __init__(self,
                 encoder_path: str,
                 decoder_path: str,
                 freeze_bert: bool = True,
                 pad_token_id: int = 128256,
                 bert_token_id: int = 128257,
                 bert_final_seq_len: int = 512):
        super().__init__()
        self.bert_final_seq_len = bert_final_seq_len  # 最终BERT序列长度限制（未实际使用，可考虑删除或后续优化）
        # Final BERT sequence length limit (not actually used; consider removing or optimizing later)

        # --- Load BERT/RoBERTa Encoder  ---
        # --- 加载 BERT/RoBERTa 编码器  ---
        print(f"Loading BERT/RoBERTa encoder from: {encoder_path}...")
        try:
            # Load the BERT-style encoder model.
            # 加载 BERT 风格的编码器模型。
            self.bert_encoder = AutoModel.from_pretrained(encoder_path)
            encoding_dim = self.bert_encoder.config.hidden_size
            print(f"BERT/RoBERTa loaded . Hidden size: {encoding_dim}")
        except Exception as e:
            print(f"Error loading BERT/RoBERTa model: {e}")
            raise e

        # Freeze BERT parameters if specified.
        # 如果指定，冻结 BERT 参数。
        self.freeze_bert = freeze_bert
        if freeze_bert:
            for param in self.bert_encoder.parameters():
                param.requires_grad = False
            self.bert_encoder.eval()
            print("BERT/RoBERTa parameters are frozen.")
        else:
            print("BERT/RoBERTa parameters are trainable.")

        # --- Load Llama model  ---
        # --- 加载 Llama 模型  ---
        print(f"Loading Llama base model from: {decoder_path}...")
        try:
            # Load the causal language model (LLM).
            # 加载因果语言模型（LLM）。
            self.llama = AutoModelForCausalLM.from_pretrained(decoder_path)
            # Freeze Llama's parameters.
            # 冻结 Llama 的参数。
            for param in self.llama.parameters():
                param.requires_grad = False
            self.llama.eval()
            self.llama_dim = self.llama.config.hidden_size
            self.pad_token_id = pad_token_id  # pad token ID 
            self.bert_token_id = bert_token_id
            # bert_token_id indicate the token_id of <BERT>, which indicate the place where bert encoded content be inserted into llama query, similar like <image> in LLaVA
            # This is the pad token ID for  Llama-3 (self-defined)
            print(f"Llama loaded . Hidden size: {self.llama_dim}")
        except Exception as e:
            print(f"Error loading Llama model: {e}")
            raise e

        # --- Initialize custom modules ---
        # --- 初始化自定义模块  ---
        print("Initializing custom modules (MLPs, AttentionPool)...")
        self.mlp1 = nn.Sequential(
            # Project BERT's hidden size to Llama's hidden size.
            # 将 BERT 的隐藏层维度投影到 Llama 的隐藏层维度。
            nn.Linear(encoding_dim, self.llama_dim),
            nn.ReLU(),
            nn.Linear(self.llama_dim, self.llama_dim)
        )
        # 确保 MLP 参数可训练
        # Ensure MLP parameters are trainable
        for param in self.mlp1.parameters():
            param.requires_grad = True

        # Attention pooling layers to reduce sequence length.
        # 注意力池化层，用于缩减序列长度。
        self.stage1 = AttentionPoolLayer(encoding_dim, pool_ratio=2)
        self.stage2 = AttentionPoolLayer(encoding_dim, pool_ratio=2)
        # 确保注意力池化层参数可训练
        # Ensure attention pooling layers' parameters are trainable
        for param in self.stage1.parameters():
            param.requires_grad = True
        for param in self.stage2.parameters():
            param.requires_grad = True

    def forward(self, 
                encoder_input, 
                encoder_input_atten, 
                question_ids, 
                question_attention_mask, 
                labels):
        """
        前向传播函数
        Forward pass of the model
        """
        batch_size = question_ids.shape[0]

        # --- 1. BERT Encoding ---
        # 使用 BERT-style encoder-only model 编码输入文本 (history context)
        # Encode input text (history context) using BERT-style encoder-only model
        bert_outputs = self.bert_encoder(
            input_ids=encoder_input,
            attention_mask=encoder_input_atten,
            return_dict=True
        )
        encoding_emb = bert_outputs.last_hidden_state  # [B, L_bert=bert_final_seq_len, D_bert]
        # BERT 最后一层隐藏状态

        # --- 2. Pooling (Left-Padding Aware with Per-Sample Processing) ---
        # 处理左填充（ padding_side="left"）
        # Handle left-padding (tokenizer uses padding_side="left")
        effective_lengths = encoder_input_atten.sum(dim=1)  # 每个样本的真实长度 real length for each sample
        max_seq_len = encoder_input_atten.shape[1]         # 最大序列长度 max sequence length
        
        stage1_pooled_emb_list = []
        stage1_pooled_len = []
        
        # 逐样本处理以支持变长序列
        # Process each sample individually to support variable-length sequences
        for i in range(batch_size):
            sample_len = effective_lengths[i].item()
            start_idx = max_seq_len - sample_len  # 左填充时真实内容起始位置 left padding content real start position
            sample_emb = encoding_emb[i, start_idx : start_idx + sample_len]  # [L_i, D]

            # 如果长度为奇数，保留第一个 token，其余两两分组
            # If length is odd, keep first token, group rest in pairs
            if sample_len % 2 != 0:
                odd_token = sample_emb[0:1]  # [1, D]
                # 将剩余部分 reshape 成 (N_pairs, 2, D)  reshape the rest into (N_pairs, 2, D) 
                reshaped_emb = sample_emb[1:].view(-1, 2, self.bert_encoder.config.hidden_size)
                # 计算注意力权重：对每对中的两个向量打分 conpute attention weight via attention pooling stage1
                weights = torch.softmax(self.stage1.query(reshaped_emb), dim=1)  # [N_pairs, 2, 1]
                # 加权求和：每对输出一个融合向量 weighted sum: output a confused tensor with each group
                pooled_emb = torch.sum(reshaped_emb * weights, dim=1)  # [N_pairs, D]
                # 拼接保留的奇数 token    concat the remained odd token
                final_pooled_emb = torch.cat([odd_token, pooled_emb], dim=0)  # [N_pairs+1, D]
            else:
                reshaped_emb = sample_emb.view(-1, 2, self.bert_encoder.config.hidden_size)
                weights = torch.softmax(self.stage1.query(reshaped_emb), dim=1)
                final_pooled_emb = torch.sum(reshaped_emb * weights, dim=1)  # [N_pairs, D]
            
            stage1_pooled_emb_list.append(final_pooled_emb)
            stage1_pooled_len.append(final_pooled_emb.shape[0])

        # 第二阶段池化：进一步压缩序列
        # Second-stage pooling: further compress sequence
        stage2_pooled_emb_list = []
        final_effective_lengths = torch.zeros(batch_size, dtype=torch.int64, device=encoding_emb.device)

        for i in range(batch_size):
            sample_emb = stage1_pooled_emb_list[i]  # 上一阶段输出 output in stage1
            sample_len = stage1_pooled_len[i]

            if sample_len % 2 != 0:
                odd_token = sample_emb[0:1]
                reshaped_emb = sample_emb[1:].view(-1, 2, self.bert_encoder.config.hidden_size)
                weights = torch.softmax(self.stage2.query(reshaped_emb), dim=1)
                pooled_emb = torch.sum(reshaped_emb * weights, dim=1)
                final_pooled_emb = torch.cat([odd_token, pooled_emb], dim=0)
            else:
                reshaped_emb = sample_emb.view(-1, 2, self.bert_encoder.config.hidden_size)
                weights = torch.softmax(self.stage2.query(reshaped_emb), dim=1)
                pooled_emb = torch.sum(reshaped_emb * weights, dim=1)
                final_pooled_emb = pooled_emb

            stage2_pooled_emb_list.append(final_pooled_emb)
            final_effective_lengths[i] = final_pooled_emb.shape[0]  # 记录最终有效长度 record the final effective length

        # --- 3. MLP and Reshaping ---
        # 必须在 MLP 之前进行 padding，因为 MLP 期望固定维度输入
        # Must pad before MLP since MLP expects fixed-dim input
        x = pad_sequence(stage2_pooled_emb_list, batch_first=True, padding_value=0.0, padding_side="left")
        # x: [B, L_pooled_max, D_bert]

        # 将 pooled 后的特征投影到 Llama 的 embedding 空间
        # Project pooled features into Llama's embedding space
        x = self.mlp1(x.view(-1, x.shape[-1]))  # [B * L_pooled_max, D_llama]
        history_prompt_embeds = x.view(batch_size, -1, self.llama_dim)  # [B, L_pooled_max, D_llama]

        # --- 4. Get Question Embedding ---
        # 获取问题的原始嵌入（不经过 BERT）
        # Get raw embeddings of the question (without BERT)
        decoding_emb = self.llama.get_input_embeddings()(question_ids)  # [B, L_q, D_llama]

        # --- 5. Prepare Labels and Attention Masks ---
        # 将 pad token 替换为 -100，以便在计算 loss 时忽略
        # Replace pad token with -100 so it's ignored during loss computation
        labels[labels == self.pad_token_id] = -100 
        
        # 找到插入位置：存在特殊 token 128257 <BERT> 作为插入锚点
        # Find insertion position: special token 128257 <BERT> is the insertion anchor
        match_mask = (question_ids == self.bert_token_id)
        if not match_mask.any(dim=1).all():
            raise ValueError(" Error: Target token_id not found in all samples.")
        
        insert_pos = match_mask.float().argmax(dim=1)  # 每个样本中第一个匹配位置

        # 分割问题嵌入和 mask：插入点前 & 插入点后
        # Split question embeddings and masks: before & after insertion point
        before_emb = []
        after_emb = []
        before_mask = []
        after_mask = []
        
        for b in range(batch_size):
            idx = insert_pos[b].item()
            before_emb.append(decoding_emb[b, :idx])           # [L_before, D]
            after_emb.append(decoding_emb[b, idx + 1:])        # [L_after, D]
            before_mask.append(question_attention_mask[b, :idx])   # [L_before]
            after_mask.append(question_attention_mask[b, idx + 1:]) # [L_after]

        # --- 6. Concatenate Embeddings and Masks ---
        # 处理 history_prompt_embeds 的左填充问题（与 before_emb 拼接前对齐）
        # Handle left-padding of history_prompt_embeds before concatenation
        valid_history_prompt_embeds = []
        valid_history_mask = []
        max_history_len = final_effective_lengths.max().item()

        for i in range(batch_size):
            valid_len = final_effective_lengths[i].item()
            start_idx = max_history_len - valid_len  # 左填充偏移 left padding move
            valid_history_prompt_embeds.append(history_prompt_embeds[i, start_idx:])  # 取有效部分 only use the effective part
            valid_history_mask.append(torch.ones(valid_len, dtype=torch.int64, device=history_prompt_embeds.device))

        # 构造最终输入：[before][history][after]
        # Construct final input: [before][history][after]
        combined_embeddings_list = [
            torch.cat([before_emb[b], valid_history_prompt_embeds[b], after_emb[b]], dim=0)
            for b in range(batch_size)
        ]
        combined_masks_list = [
            torch.cat([before_mask[b], valid_history_mask[b], after_mask[b]], dim=0)
            for b in range(batch_size)
        ]

        # 使用 pad_sequence 安全地填充为统一长度，保留梯度流
        # Safely pad to uniform length using pad_sequence, preserving gradient flow
        inputs_embeds = pad_sequence(combined_embeddings_list, batch_first=True, padding_value=0.0, padding_side="left")
        final_attention_mask = pad_sequence(combined_masks_list, batch_first=True, padding_value=0, padding_side="left")

        # --- 7. Prepare Labels and Pass to the LLM Decoder ---
        # 调整 labels 位置，使其对应最终输出序列的 answer 部分
        # Adjust labels to align with answer portion in final output sequence
        if labels is not None:
            input_len = inputs_embeds.shape[1]
            answer_len = labels.shape[1]
            # 创建全 -100 的 label tensor
            # build label tensor 
            adjusted_labels = torch.full((batch_size, input_len), -100, dtype=labels.dtype, device=labels.device)
            # 将 answer labels 放在最后
            # left padding, answer labels in the last
            adjusted_labels[:, -answer_len:] = labels
            final_labels = adjusted_labels
        else:
            final_labels = None
            
        # 将拼接后的嵌入输入 Llama 模型进行解码
        # Feed concatenated embeddings into Llama for decoding
        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=final_attention_mask,
            labels=final_labels,
            return_dict=True
        )

        # 对 loss 取均值（如果存在）
        # Take mean of loss if exists
        if outputs.loss is not None:
            outputs.loss = outputs.loss.mean()
        
        return outputs
    



    @torch.no_grad()
    def generate_answer(self,encoder_input,encoder_input_atten,question_ids,question_attention_mask,max_new_tokens=20):
        self.eval()
        batch_size = question_ids.shape[0]

        # --- 1. BERT Encoding ---
        # The BERT part of the code is correct.
        bert_outputs = self.bert_encoder(
            input_ids=encoder_input,
            attention_mask=encoder_input_atten,
            return_dict=True
        )
        encoding_emb = bert_outputs.last_hidden_state

        # --- 2. Pooling (Left-Padding Aware with Per-Sample Processing) ---
        # This per-sample processing is necessary for varying sequence lengths.
        effective_lengths = encoder_input_atten.sum(dim=1)
        max_seq_len = encoder_input_atten.shape[1]
        
        stage1_pooled_emb_list = []
        stage1_pooled_len = []
        
        for i in range(batch_size):
            sample_len = effective_lengths[i].item()
            start_idx = max_seq_len - sample_len
            sample_emb = encoding_emb[i, start_idx : start_idx + sample_len]

            if sample_len % 2 != 0:
                odd_token = sample_emb[0:1]
                reshaped_emb = sample_emb[1:].view(-1, 2, self.bert_encoder.config.hidden_size)
                weights = torch.softmax(self.stage1.query(reshaped_emb), dim=1)
                pooled_emb = torch.sum(reshaped_emb * weights, dim=1)
                final_pooled_emb = torch.cat([odd_token, pooled_emb], dim=0)
            else:
                reshaped_emb = sample_emb.view(-1, 2, self.bert_encoder.config.hidden_size)
                weights = torch.softmax(self.stage1.query(reshaped_emb), dim=1)
                final_pooled_emb = torch.sum(reshaped_emb * weights, dim=1)
            
            stage1_pooled_emb_list.append(final_pooled_emb)
            stage1_pooled_len.append(final_pooled_emb.shape[0])

        stage2_pooled_emb_list = []
        final_effective_lengths = torch.zeros(batch_size, dtype=torch.int64, device=encoding_emb.device)

        for i in range(batch_size):
            sample_emb = stage1_pooled_emb_list[i]
            sample_len = stage1_pooled_len[i]

            if sample_len % 2 != 0:
                odd_token = sample_emb[0:1]
                reshaped_emb = sample_emb[1:].view(-1, 2, self.bert_encoder.config.hidden_size)
                weights = torch.softmax(self.stage2.query(reshaped_emb), dim=1)
                pooled_emb = torch.sum(reshaped_emb * weights, dim=1)
                final_pooled_emb = torch.cat([odd_token, pooled_emb], dim=0)
            else:
                reshaped_emb = sample_emb.view(-1, 2, self.bert_encoder.config.hidden_size)
                weights = torch.softmax(self.stage2.query(reshaped_emb), dim=1)
                pooled_emb = torch.sum(reshaped_emb * weights, dim=1)
                final_pooled_emb = pooled_emb

            stage2_pooled_emb_list.append(final_pooled_emb)
            final_effective_lengths[i] = final_pooled_emb.shape[0]

        # --- 3. MLP and Reshaping ---
        # We must pad the list of tensors BEFORE passing to the MLP
        # 使用 pad_sequence 安全地填充张量列表，保留梯度流
        # use pad_sequence to maintain the gradient flow
        x = pad_sequence(stage2_pooled_emb_list, batch_first=True, padding_value=0.0, padding_side="left")
        # The MLP takes the padded tensor as input
        x = self.mlp1(x.view(-1, x.shape[-1]))
        history_prompt_embeds = x.view(batch_size, -1, self.llama_dim)
        
        # --- 4. Get Question Embedding ---
        decoding_emb = self.llama.get_input_embeddings()(question_ids)
        
        
        match_mask = (question_ids == self.bert_token_id)
        if not match_mask.any(dim=1).all():
            raise ValueError("Error: Target token_id not found in all samples.")
        
        insert_pos = match_mask.float().argmax(dim=1)
        
        before_emb = []
        after_emb = []
        before_mask = []
        after_mask = []
        
        for b in range(batch_size):
            idx = insert_pos[b].item()
            before_emb.append(decoding_emb[b, :idx])
            after_emb.append(decoding_emb[b, idx + 1:])
            before_mask.append(question_attention_mask[b, :idx])
            after_mask.append(question_attention_mask[b, idx + 1:])

        # --- 6. Concatenate Embeddings and Masks ---
        valid_history_prompt_embeds = []
        valid_history_mask = []
        max_history_len = final_effective_lengths.max().item()

        for i in range(batch_size):
            valid_len = final_effective_lengths[i].item()
            start_idx = max_history_len - valid_len
            valid_history_prompt_embeds.append(history_prompt_embeds[i, start_idx:])
            valid_history_mask.append(torch.ones(valid_len, dtype=torch.int64, device=history_prompt_embeds.device))

        combined_embeddings_list = [
            torch.cat([before_emb[b], valid_history_prompt_embeds[b], after_emb[b]], dim=0)
            for b in range(batch_size)
        ]
        combined_masks_list = [
            torch.cat([before_mask[b], valid_history_mask[b], after_mask[b]], dim=0)
            for b in range(batch_size)
        ]

        # 使用 pad_sequence 进行最终的填充，保留梯度流
        # Pad the embeddings and masks using pad_sequence to preserve the gradient.
        inputs_embeds = pad_sequence(combined_embeddings_list, batch_first=True, padding_value=0.0,padding_side="left")
        final_attention_mask = pad_sequence(combined_masks_list, batch_first=True, padding_value=0,padding_side="left")
        

        # 7. 生成
        outputs = self.llama.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=final_attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id = self.pad_token_id,
            eos_token_id = 128009,
        )

        return outputs
    




class UBC_model(nn.Module):
    def __init__(self,
                 encoder_path: str,
                 decoder_path: str,
                 lora_r: int,
                 lora_alpha: int,
                 codebook_path:str,
                 lora_target_modules: list,
                 lora_dropout: float,
                 lora_bias: str,
                 freeze_bert: bool = True,
                 pad_token_id: int = 128256,
                 bert_token_id: int = 128257,
                 bert_final_seq_len: int = 256):
        super().__init__()
        self.bert_final_seq_len = bert_final_seq_len

        # --- Load BERT/RoBERTa Encoder  ---
        # --- 加载 BERT/RoBERTa 编码器  ---
        print(f"Loading BERT/RoBERTa encoder from: {encoder_path}...")
        try:
            # Load the BERT-style encoder model.
            # 加载 BERT 风格的编码器模型。
            self.bert_encoder = AutoModel.from_pretrained(encoder_path)
            encoding_dim = self.bert_encoder.config.hidden_size
            print(f"BERT/RoBERTa loaded . Hidden size: {encoding_dim}")
        except Exception as e:
            print(f"Error loading BERT/RoBERTa model: {e}")
            raise e

        # Freeze BERT parameters if specified.
        # 如果指定，冻结 BERT 参数。
        self.freeze_bert = freeze_bert
        if freeze_bert:
            for param in self.bert_encoder.parameters():
                param.requires_grad = False
            self.bert_encoder.eval()
            print("BERT/RoBERTa parameters are frozen.")
        else:
            print("BERT/RoBERTa parameters are trainable.")

        # --- Load Llama model  ---
        # --- 加载 Llama 模型  ---
        print(f"Loading Llama base model from: {decoder_path}...")
        try:
            base_model = AutoModelForCausalLM.from_pretrained(decoder_path)
            for param in base_model.parameters():
                param.requires_grad = False
            base_model.eval()
            self.llama_dim = base_model.config.hidden_size

            self.pad_token_id = pad_token_id
            self.bert_token_id = bert_token_id
            # bert_token_id indicate the token_id of <BERT>, which indicate the place where bert encoded content be inserted into llama query, similar like <image> in LLaVA
            # This is the pad token ID for  Llama-3 (self-defined)
            print(f"Llama loaded . Hidden size: {self.llama_dim}")
        except Exception as e:
            print(f"Error loading Llama model: {e}")
            raise e

        # --- Initialize custom modules  ---
        # --- 初始化自定义模块  ---
        print("Initializing custom modules (MLPs, AttentionPool)...")
        self.mlp1 = nn.Sequential(
            # Project BERT's hidden size to Llama's hidden size.
            # 将 BERT 的隐藏层维度投影到 Llama 的隐藏层维度。
            nn.Linear(encoding_dim, self.llama_dim),
            nn.ReLU(),
            nn.Linear(self.llama_dim, self.llama_dim)
        )
        # 确保 MLP 参数冻结
        # Ensure MLP parameters are frozen
        for param in self.mlp1.parameters():
                param.requires_grad = False
        # 移除 .half()
        self.codebook =  CodebookAttention(codebook_path=codebook_path, num_heads=2, dropout=0.1, freeze=False) # initialize codebook 初始化codebook

        # Attention pooling layers to reduce sequence length.
        # 注意力池化层，用于缩减序列长度。
        self.stage1 = AttentionPoolLayer(encoding_dim, pool_ratio=2)
        self.stage2 = AttentionPoolLayer(encoding_dim, pool_ratio=2)
        # 确保注意力池化层参数冻结
        # Ensure attention pooling layers' parameters are trainable
        for param in self.stage1.parameters():
            param.requires_grad = False
        for param in self.stage2.parameters():
            param.requires_grad = False
        # --- 应用 LoRA ---
        # --- Apply LoRA ---
        print("Apply LoRA Config...")
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, target_modules=lora_target_modules,
            lora_dropout=lora_dropout, bias=lora_bias, task_type=TaskType.CAUSAL_LM
        )
        self.llama = get_peft_model(base_model, lora_config)

        self.llama.train() 
        # 使 LoRA 层和任何未冻结的部分（如 lm_head）处于训练模式
        # make LoRA and unfrozen part(like lm_head) in training mode
        self.llama.base_model.eval() 

        self.llama.print_trainable_parameters()

    def _apply_two_stage_pooling(self, padded_seq, effective_lengths, max_seq_len):
            batch_size = padded_seq.shape[0]
            device = padded_seq.device

            stage1_list = []
            for i in range(batch_size):
                L = effective_lengths[i].item()
                start = max_seq_len - L
                seq = padded_seq[i, start:start+L]
                # Stage 1
                if L % 2 != 0:
                    odd = seq[0:1]
                    pairs = seq[1:].view(-1, 2, seq.size(-1))
                    w = torch.softmax(self.stage1.query(pairs), dim=1)
                    pooled = (pairs * w).sum(dim=1)
                    out = torch.cat([odd, pooled], dim=0)
                else:
                    pairs = seq.view(-1, 2, seq.size(-1))
                    w = torch.softmax(self.stage1.query(pairs), dim=1)
                    out = (pairs * w).sum(dim=1)
                stage1_list.append(out)

            stage2_list = []
            final_lengths = []
            for out in stage1_list:
                L = out.shape[0]
                if L % 2 != 0:
                    odd = out[0:1]
                    pairs = out[1:].view(-1, 2, out.size(-1))
                    w = torch.softmax(self.stage2.query(pairs), dim=1)
                    pooled = (pairs * w).sum(dim=1)
                    final = torch.cat([odd, pooled], dim=0)
                else:
                    pairs = out.view(-1, 2, out.size(-1))
                    w = torch.softmax(self.stage2.query(pairs), dim=1)
                    final = (pairs * w).sum(dim=1)
                stage2_list.append(final)
                final_lengths.append(final.shape[0])

            padded = pad_sequence(stage2_list, batch_first=True, padding_value=0.0, padding_side='left')
            return padded, torch.tensor(final_lengths, device=device)
    


    def forward(self, bert_profile_id, bert_profile_atten, bert_qa_id, bert_qa_atten,
                answer_start_indices, question_ids, question_attention_mask, labels,mode='original',num_finite=4):
        """
        前向传播函数，支持任意数量的 QA 对
        Forward pass of the model, supporting an arbitrary number of QA pairs
        """

        batch_size, qa_num, _ = bert_qa_id.shape

        # --- 1. BERT Encoding ---
        # 使用 BERT-style encoder-only model 编码输入文本 (history context)
        # Encode input text (history context) using BERT-style encoder-only model
        bert_forward_context = torch.no_grad() if self.freeze_bert else torch.enable_grad()
        with bert_forward_context:
            # 编码 profile
            # Encode profile
            bert_profile = self.bert_encoder(
                input_ids=bert_profile_id,
                attention_mask=bert_profile_atten,
                return_dict=True
            )
            # 编码所有 qa 对。为了并行处理，将 batch 和 qa_num 维度合并。
            # Encode all qa pairs. Merge batch and qa_num dimensions for parallel processing.
            bert_qa_outputs = self.bert_encoder(
                input_ids=bert_qa_id.view(-1, bert_qa_id.size(-1)),
                attention_mask=bert_qa_atten.view(-1, bert_qa_atten.size(-1)),
                return_dict=True
            )

        # 从 BERT 模型获取最终隐藏状态
        # Get the last hidden states from the BERT model
        encoding_profile = bert_profile.last_hidden_state           # [B, L_p, D_bert]
        encoding_qa = bert_qa_outputs.last_hidden_state.view(batch_size, qa_num, -1, self.bert_encoder.config.hidden_size) # [B, N_qa, L_q, D_bert]

        # 计算每个序列的实际长度（非填充部分）
        # Calculate actual lengths (non-padded parts) of each sequence
        len_profile = torch.sum(bert_profile_atten, dim=1).long()    # [B]
        len_qa = torch.sum(bert_qa_atten, dim=2).long()              # [B, N_qa]

        # 提取非填充部分（左填充时取最后若干个 token）
        # Extract non-padded parts (take last L tokens for left padding)
        non_padded_profile = [encoding_profile[i, -len_profile[i]:] for i in range(batch_size)]
        non_padded_qa = [[encoding_qa[i, j, -len_qa[i, j]:] for j in range(qa_num)] for i in range(batch_size)]

        # 将 profile + 所有 qa 的非填充编码拼接在一起
        # Concatenate non-padded encodings of profile + all qas
        encoding_list = []
        for i in range(batch_size):
            # 创建一个包含 profile 编码的列表
            # Create a list containing the profile encoding
            seq_list = [non_padded_profile[i]]
            # 遍历所有 qa 对并添加到列表中
            # Iterate through all qa pairs and add them to the list
            for j in range(qa_num):
                seq_list.append(non_padded_qa[i][j])
            # 拼接列表中的所有张量
            # Concatenate all tensors in the list
            encoding_list.append(torch.cat(seq_list, dim=0))

        # 将可变长度序列 pad 成固定长度张量
        # Pad variable-length sequences into fixed-length tensor
        padded_encoding = pad_sequence(encoding_list, batch_first=True,padding_value=0.0, padding_side="left")  # [B, max_L_total, D_bert]

        # --- 2. 构建答案相关的编码序列 (profile + 所有答案) ---
        # --- 2. Construct answer-related encoding (profile + all answers) ---
        answer_encodings_list = []
        answer_encodings_list_len = []
        for i in range(batch_size):
            # 创建一个包含 profile 编码的列表
            # Create a list containing the profile encoding
            seq_list = [non_padded_profile[i]]
            # 遍历所有 qa 对的答案部分并添加到列表中
            # Iterate through all qa pairs' answer spans and add them to the list
            for j in range(qa_num):
                # 获取答案起始位置
                # Get the answer start index
                start_qa = answer_start_indices[i, j].item()
                # 从非填充的编码中提取答案部分
                # Extract the answer span from the non-padded embedding
                answer_span = non_padded_qa[i][j][start_qa:]  # [L_aj, D_bert]
                seq_list.append(answer_span)
            
            # 拼接 profile 和所有答案
            # Concatenate profile and all answers
            concatenated_answer = torch.cat(seq_list, dim=0)
            answer_encodings_list.append(concatenated_answer)
            answer_encodings_list_len.append(len(concatenated_answer))

        # 将答案编码序列 pad 成统一长度
        # Pad answer encodings to same length
        padded_answer = pad_sequence(answer_encodings_list, batch_first=True,padding_value=0.0, padding_side="left")  # [B, max_L_a, D_bert]

        # --- 3. 两阶段注意力池化（Two-stage attention pooling）---
        # --- 3. Two-stage attention pooling ---
        max_seq_len_qa = padded_encoding.shape[1]
        max_seq_len_a = padded_answer.shape[1]

        # 计算每条样本的有效长度（真实 token 数）
        # Effective lengths (number of real tokens per sample)
        effective_lengths_qa = len_profile + torch.sum(len_qa, dim=1)  # [B]
        effective_lengths_a = torch.tensor(answer_encodings_list_len, dtype=torch.long, device=padded_answer.device)  # [B]
        
        x_qa, final_effective_lengths_qa = self._apply_two_stage_pooling(padded_encoding, effective_lengths_qa, max_seq_len_qa)
        x_a, final_effective_lengths_a = self._apply_two_stage_pooling(padded_answer, effective_lengths_a, max_seq_len_a)
        
        # --- 4. MLP + Codebook + MLP ---
        # (此部分与原代码相同，无需修改)
        # (This part is the same as the original code and does not need to be modified)
        for name, param in self.codebook.named_parameters():
            _ = param

        user_model = self.codebook(x_a, final_effective_lengths_a, mode=mode, num_finite=num_finite) 

        # 构造最终输入序列：[QA pooled] + [User Model]
        # construct final input sequence: [QA pooled] + [User Model]
        x_list = []
        x_list_len = torch.zeros(batch_size, dtype=torch.int64, device=padded_answer.device)

        # 根据不同的模式处理 user_model
        # deal with user model based on mode
        if mode == "original":
            # 模式 "original": 输出维度与输入 x_a 相同，需要去除填充部分
            # mode original： exclude padding, same shape as x_a
            for i in range(batch_size):
                effective_len_qa = final_effective_lengths_qa[i]
                effective_len_a = final_effective_lengths_a[i]

                if effective_len_a > 0:
                    # 去除左填充部分  exclude left pad
                    user_seq = user_model[i, -effective_len_a:, :]  # [effective_len_a, D_codebook]
                else:
                    user_seq = user_model.new_empty(0, user_model.size(-1))

                # 获取 QA 序列的有效部分  get effective parts
                qa_seq = x_qa[i, -effective_len_qa:]

                # 拼接：QA 表示 + 用户建模结果   concat qa_seq + user model
                combined = torch.cat([qa_seq, user_seq], dim=0)
                x_list.append(combined)
                x_list_len[i] = len(combined)

        elif mode == "soft_prompt":
            # 模式 "soft_prompt": user_model 输出为单个向量 (B, D)，直接使用
            # mode soft prompt : user model is a single tensor (B,D)
            for i in range(batch_size):
                effective_len_qa = final_effective_lengths_qa[i]
                
                
                user_vector = user_model[i].unsqueeze(0)  # [1, D_codebook]

                
                qa_seq = x_qa[i, -effective_len_qa:]

                
                combined = torch.cat([qa_seq, user_vector], dim=0)
                x_list.append(combined)
                x_list_len[i] = len(combined)

        elif mode == "finite":
            # 模式 "finite": user_model 输出为 (B, num, D)，直接使用
            # mode finiete: user model output a (B,num,D) tensor, D is given
            for i in range(batch_size):
                effective_len_qa = final_effective_lengths_qa[i]
                
                
                user_seq = user_model[i] # [num, D_codebook]

                
                qa_seq = x_qa[i, -effective_len_qa:]

                
                combined = torch.cat([qa_seq, user_seq], dim=0)
                x_list.append(combined)
                x_list_len[i] = len(combined)
            
        #pad again
        x = pad_sequence(x_list, batch_first=True, padding_value=0.0, padding_side="left")


        x = self.mlp1(x)

        for idx, layer in enumerate(self.mlp1):
            if isinstance(layer, nn.Linear):
                _ = layer.weight
                _ = layer.bias

        history_prompt_embeds = x.view(batch_size, -1, self.llama_dim)

        # --- 5. 获取问题嵌入 ---
        # --- 5. Get question embeddings ---
        decoding_emb = self.llama.get_input_embeddings()(question_ids)

        labels[labels == self.pad_token_id] = -100

        # --- 6. 构造 LLaMA 输入：在特定 token 处插入历史 prompt ---
        # --- 6. Construct LLaMA input: insert history prompt at special token ---
        match_mask = (question_ids == self.bert_token_id)
        if not match_mask.any(dim=1).all():
            raise ValueError("some sample do not have <BERT>")

        insert_pos = match_mask.float().argmax(dim=1)

        before_emb = []
        after_emb = []
        before_mask = []
        after_mask = []
        for b in range(batch_size):
            idx = insert_pos[b].item()
            before_emb.append(decoding_emb[b, :idx])
            after_emb.append(decoding_emb[b, idx + 1:])
            before_mask.append(question_attention_mask[b, :idx])
            after_mask.append(question_attention_mask[b, idx + 1:])

        valid_history_prompt_embeds = []
        valid_history_mask = []
        max_history_len = x_list_len.max().item()
        for i in range(batch_size):
            valid_len = x_list_len[i].item()
            start_idx = max_history_len - valid_len
            valid_history_prompt_embeds.append(history_prompt_embeds[i, start_idx:])
            valid_history_mask.append(torch.ones(valid_len, dtype=torch.int64, device=history_prompt_embeds.device))

        combined_embeddings_list = [
            torch.cat([before_emb[b], valid_history_prompt_embeds[b], after_emb[b]], dim=0)
            for b in range(batch_size)
        ]
        combined_masks_list = [
            torch.cat([before_mask[b], valid_history_mask[b], after_mask[b]], dim=0)
            for b in range(batch_size)
        ]

        inputs_embeds = pad_sequence(combined_embeddings_list, batch_first=True, padding_value=0.0, padding_side="left")
        final_attention_mask = pad_sequence(combined_masks_list, batch_first=True, padding_value=0, padding_side="left")

        if labels is not None:
            input_len = inputs_embeds.shape[1]
            answer_len = labels.shape[1]
            adjusted_labels = torch.full((batch_size, input_len), -100, dtype=labels.dtype, device=labels.device)
            adjusted_labels[:, -answer_len:] = labels
            final_labels = adjusted_labels
        else:
            final_labels = None

        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=final_attention_mask,
            labels=final_labels,
            return_dict=True
        )

        if outputs.loss is not None:
            total_loss = outputs.loss
            outputs.loss = total_loss

        return outputs

    @torch.no_grad()
    def generate_answer(self, bert_profile_id, bert_profile_atten, bert_qa_id, bert_qa_atten,
                        answer_start_indices, question_ids, question_attention_mask, 
                        max_new_tokens=128, mode='original', num_finite=4):
        """
        Generates an answer using the multimodal model.
        使用多模态模型生成答案。

        Args:
            bert_profile_id (Tensor): Token IDs for the user profile.
                                    用户画像的 token ID。
            bert_profile_atten (Tensor): Attention mask for the user profile.
                                        用户画像的 attention mask。
            bert_qa_id (Tensor): Token IDs for multiple QA pairs.
                                多个问答对的 token ID。
            bert_qa_atten (Tensor): Attention masks for multiple QA pairs.
                                    多个问答对的 attention mask。
            answer_start_indices (Tensor): Starting indices of answers in each QA pair.
                                        每个问答对中答案的起始索引。
            question_ids (Tensor): Token IDs for the question prompt.
                                问题提示的 token ID。
            question_attention_mask (Tensor): Attention mask for the question prompt.
                                            问题提示的 attention mask。
            max_new_tokens (int): Maximum tokens to generate.
                                最大生成 token 数。
            mode (str): The operational mode for the codebook attention.
                        码本注意力的操作模式。
            num_finite (int): The number of groups for the 'finite' mode.
                            'finite' 模式下的分组数。
        
        Returns:
            Tensor: Generated token IDs.
                    生成的 token ID。
        """
        self.eval()
        batch_size, qa_num, _ = bert_qa_id.shape
        
        # --- 1. BERT Encoding ---
        # 使用 BERT-style encoder-only model 编码输入文本
        # Encode input text using BERT-style encoder-only model
        with torch.no_grad():
            bert_profile = self.bert_encoder(
                input_ids=bert_profile_id,
                attention_mask=bert_profile_atten,
                return_dict=True
            )
            
            # 编码所有 qa 对。为了并行处理，将 batch 和 qa_num 维度合并。
            # Encode all qa pairs. Merge batch and qa_num dimensions for parallel processing.
            bert_qa_outputs = self.bert_encoder(
                input_ids=bert_qa_id.view(-1, bert_qa_id.size(-1)),
                attention_mask=bert_qa_atten.view(-1, bert_qa_atten.size(-1)),
                return_dict=True
            )

        # 从 BERT 模型获取最终隐藏状态
        # Get the last hidden states from the BERT model
        encoding_profile = bert_profile.last_hidden_state  # [B, L_p, D_bert]
        # 将 QA 编码重新 reshape 回原始形状
        # Reshape QA embeddings back to original shape
        encoding_qa = bert_qa_outputs.last_hidden_state.view(batch_size, qa_num, -1, self.bert_encoder.config.hidden_size) # [B, N_qa, L_q, D_bert]

        # 计算每个序列的实际长度（非填充部分）
        # Calculate actual lengths (non-padded parts) of each sequence
        len_profile = torch.sum(bert_profile_atten, dim=1).long()  # [B]
        len_qa = torch.sum(bert_qa_atten, dim=2).long()            # [B, N_qa]

        # 提取非填充部分（左填充时取最后若干个 token）
        # Extract non-padded parts (take last L tokens for left padding)
        non_padded_profile = [encoding_profile[i, -len_profile[i]:] for i in range(batch_size)]
        non_padded_qa = [[encoding_qa[i, j, -len_qa[i, j]:] for j in range(qa_num)] for i in range(batch_size)]

        # 将 profile + 所有 qa 的非填充编码拼接在一起
        # Concatenate non-padded encodings of profile + all qas
        encoding_list = []
        for i in range(batch_size):
            seq_list = [non_padded_profile[i]]
            for j in range(qa_num):
                seq_list.append(non_padded_qa[i][j])
            encoding_list.append(torch.cat(seq_list, dim=0))

        padded_encoding = pad_sequence(encoding_list, batch_first=True, padding_value=0.0, padding_side="left")

        # --- 2. 构建答案相关的编码序列 (profile + 所有答案) ---
        # --- 2. Construct answer-related encoding (profile + all answers) ---
        answer_encodings_list = []
        answer_encodings_list_len = []
        for i in range(batch_size):
            seq_list = [non_padded_profile[i]]
            for j in range(qa_num):
                start_qa = answer_start_indices[i, j].item()
                answer_span = non_padded_qa[i][j][start_qa:]
                seq_list.append(answer_span)
            
            concatenated_answer = torch.cat(seq_list, dim=0)
            answer_encodings_list.append(concatenated_answer)
            answer_encodings_list_len.append(len(concatenated_answer))

        padded_answer = pad_sequence(answer_encodings_list, batch_first=True, padding_value=0.0, padding_side="left")

        # --- 3. 两阶段注意力池化（Two-stage attention pooling）---
        # --- 3. Two-stage attention pooling ---
        max_seq_len_qa = padded_encoding.shape[1]
        max_seq_len_a = padded_answer.shape[1]

        effective_lengths_qa = len_profile + torch.sum(len_qa, dim=1)
        effective_lengths_a = torch.tensor(answer_encodings_list_len, dtype=torch.long, device=padded_answer.device)
        
        x_qa, final_effective_lengths_qa = self._apply_two_stage_pooling(padded_encoding, effective_lengths_qa, max_seq_len_qa)
        x_a, final_effective_lengths_a = self._apply_two_stage_pooling(padded_answer, effective_lengths_a, max_seq_len_a)

        # --- 4. MLP + Codebook + MLP ---
        # 使用 codebook 对用户特征进行建模，根据 mode 参数动态调整
        # Model user features using codebook, dynamically adjust based on mode
        user_model = self.codebook(x_a, final_effective_lengths_a, mode=mode, num_finite=num_finite) 

        # 构造最终输入序列：[QA pooled] + [User Model]
        # Construct final input: [QA pooled] + [User Model]
        x_list = []
        x_list_len = torch.zeros(batch_size, dtype=torch.int64, device=padded_answer.device)
        
        if mode == "original":
            for i in range(batch_size):
                effective_len_qa = final_effective_lengths_qa[i]
                effective_len_a = final_effective_lengths_a[i]

                if effective_len_a > 0:
                    user_seq = user_model[i, -effective_len_a:, :]
                else:
                    user_seq = user_model.new_empty(0, user_model.size(-1))

                qa_seq = x_qa[i, -effective_len_qa:]
                combined = torch.cat([qa_seq, user_seq], dim=0)
                x_list.append(combined)
                x_list_len[i] = len(combined)

        elif mode == "soft_prompt":
            for i in range(batch_size):
                effective_len_qa = final_effective_lengths_qa[i]
                user_vector = user_model[i].unsqueeze(0)

                qa_seq = x_qa[i, -effective_len_qa:]
                combined = torch.cat([qa_seq, user_vector], dim=0)
                x_list.append(combined)
                x_list_len[i] = len(combined)

        elif mode == "finite":
            for i in range(batch_size):
                effective_len_qa = final_effective_lengths_qa[i]
                user_seq = user_model[i]

                qa_seq = x_qa[i, -effective_len_qa:]
                combined = torch.cat([qa_seq, user_seq], dim=0)
                x_list.append(combined)
                x_list_len[i] = len(combined)
            
        x = pad_sequence(x_list, batch_first=True, padding_value=0.0, padding_side="left")
        x = self.mlp1(x)
        history_prompt_embeds = x.view(batch_size, -1, self.llama_dim)

        # --- 5. 获取问题嵌入 ---
        # --- 5. Get question embeddings ---
        decoding_emb = self.llama.get_input_embeddings()(question_ids)

        # --- 6. 构造 LLaMA 输入：在特定 token 处插入历史 prompt ---
        # --- 6. Construct LLaMA input: insert history prompt at special token ---
        match_mask = (question_ids == self.bert_token_id)
        if not match_mask.any(dim=1).all():
            raise ValueError("some sample do not have <BERT>")

        insert_pos = match_mask.float().argmax(dim=1)

        before_emb = []
        after_emb = []
        before_mask = []
        after_mask = []
        for b in range(batch_size):
            idx = insert_pos[b].item()
            before_emb.append(decoding_emb[b, :idx])
            after_emb.append(decoding_emb[b, idx + 1:])
            before_mask.append(question_attention_mask[b, :idx])
            after_mask.append(question_attention_mask[b, idx + 1:])

        valid_history_prompt_embeds = []
        valid_history_mask = []
        max_history_len = x_list_len.max().item()
        for i in range(batch_size):
            valid_len = x_list_len[i].item()
            start_idx = max_history_len - valid_len
            valid_history_prompt_embeds.append(history_prompt_embeds[i, start_idx:])
            valid_history_mask.append(torch.ones(valid_len, dtype=torch.int64, device=history_prompt_embeds.device))

        combined_embeddings_list = [
            torch.cat([before_emb[b], valid_history_prompt_embeds[b], after_emb[b]], dim=0)
            for b in range(batch_size)
        ]
        combined_masks_list = [
            torch.cat([before_mask[b], valid_history_mask[b], after_mask[b]], dim=0)
            for b in range(batch_size)
        ]

        inputs_embeds = pad_sequence(combined_embeddings_list, batch_first=True, padding_value=0.0, padding_side="left")
        final_attention_mask = pad_sequence(combined_masks_list, batch_first=True, padding_value=0, padding_side="left")

        # 7. 生成
        outputs = self.llama.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=final_attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id = self.pad_token_id,
            eos_token_id = 128009,
        )
        
        return outputs




class CSG(nn.Module):
    def __init__(self,
                 encoder_path: str,
                 decoder_path: str,
                 lora_r: int,
                 lora_alpha: int,
                 codebook_path:str,
                 lora_target_modules: list,
                 lora_dropout: float,
                 lora_bias: str,
                 freeze_bert: bool = True,
                 pad_token_id: int = 128256,
                 bert_token_id: int = 128257,
                 bert_final_seq_len: int = 256):
        super().__init__()
        self.bert_final_seq_len = bert_final_seq_len

        # --- Load BERT/RoBERTa Encoder  ---
        # --- 加载 BERT/RoBERTa 编码器  ---
        print(f"Loading BERT/RoBERTa encoder from: {encoder_path}...")
        try:
            # Load the BERT-style encoder model.
            # 加载 BERT 风格的编码器模型。
            self.bert_encoder = AutoModel.from_pretrained(encoder_path)
            encoding_dim = self.bert_encoder.config.hidden_size
            print(f"BERT/RoBERTa loaded . Hidden size: {encoding_dim}")
        except Exception as e:
            print(f"Error loading BERT/RoBERTa model: {e}")
            raise e

        # Freeze BERT parameters if specified.
        # 如果指定，冻结 BERT 参数。
        self.freeze_bert = freeze_bert
        if freeze_bert:
            for param in self.bert_encoder.parameters():
                param.requires_grad = False
            self.bert_encoder.eval()
            print("BERT/RoBERTa parameters are frozen.")
        else:
            print("BERT/RoBERTa parameters are trainable.")

        # --- Load Llama model  ---
        # --- 加载 Llama 模型  ---
        print(f"Loading Llama base model from: {decoder_path}...")
        try:
            base_model = AutoModelForCausalLM.from_pretrained(decoder_path)
            for param in base_model.parameters():
                param.requires_grad = False
            base_model.eval()
            self.llama_dim = base_model.config.hidden_size

            self.pad_token_id = pad_token_id
            self.bert_token_id = bert_token_id
            # bert_token_id indicate the token_id of <BERT>, which indicate the place where bert encoded content be inserted into llama query, similar like <image> in LLaVA
            # This is the pad token ID for  Llama-3 (self-defined)
            print(f"Llama loaded . Hidden size: {self.llama_dim}")
        except Exception as e:
            print(f"Error loading Llama model: {e}")
            raise e

        # --- Initialize custom modules  ---
        # --- 初始化自定义模块  ---
        print("Initializing custom modules (MLPs, AttentionPool)...")
        self.mlp1 = nn.Sequential(
            # Project BERT's hidden size to Llama's hidden size.
            # 将 BERT 的隐藏层维度投影到 Llama 的隐藏层维度。
            nn.Linear(encoding_dim, self.llama_dim),
            nn.ReLU(),
            nn.Linear(self.llama_dim, self.llama_dim)
        )
        # 确保 MLP 参数冻结
        # Ensure MLP parameters are frozen
        for param in self.mlp1.parameters():
                param.requires_grad = False
        # 移除 .half()
        self.codebook =  CodebookAttention(codebook_path=codebook_path, num_heads=2, dropout=0.1, freeze=False) # initialize codebook 初始化codebook

        # Attention pooling layers to reduce sequence length.
        # 注意力池化层，用于缩减序列长度。
        self.stage1 = AttentionPoolLayer(encoding_dim, pool_ratio=2)
        self.stage2 = AttentionPoolLayer(encoding_dim, pool_ratio=2)
        # 确保注意力池化层参数冻结
        # Ensure attention pooling layers' parameters are trainable
        for param in self.stage1.parameters():
            param.requires_grad = False
        for param in self.stage2.parameters():
            param.requires_grad = False
        # --- 应用 LoRA ---
        # --- Apply LoRA ---
        print("Apply LoRA Config...")
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, target_modules=lora_target_modules,
            lora_dropout=lora_dropout, bias=lora_bias, task_type=TaskType.CAUSAL_LM
        )
        self.llama = get_peft_model(base_model, lora_config)

        self.llama.train() 
        # 使 LoRA 层和任何未冻结的部分（如 lm_head）处于训练模式
        # make LoRA and unfrozen part(like lm_head) in training mode
        self.llama.base_model.eval() 

        self.llama.print_trainable_parameters()

    def _apply_two_stage_pooling(self, padded_seq, effective_lengths, max_seq_len):
            batch_size = padded_seq.shape[0]
            device = padded_seq.device

            stage1_list = []
            for i in range(batch_size):
                L = effective_lengths[i].item()
                start = max_seq_len - L
                seq = padded_seq[i, start:start+L]
                # Stage 1
                if L % 2 != 0:
                    odd = seq[0:1]
                    pairs = seq[1:].view(-1, 2, seq.size(-1))
                    w = torch.softmax(self.stage1.query(pairs), dim=1)
                    pooled = (pairs * w).sum(dim=1)
                    out = torch.cat([odd, pooled], dim=0)
                else:
                    pairs = seq.view(-1, 2, seq.size(-1))
                    w = torch.softmax(self.stage1.query(pairs), dim=1)
                    out = (pairs * w).sum(dim=1)
                stage1_list.append(out)

            stage2_list = []
            final_lengths = []
            for out in stage1_list:
                L = out.shape[0]
                if L % 2 != 0:
                    odd = out[0:1]
                    pairs = out[1:].view(-1, 2, out.size(-1))
                    w = torch.softmax(self.stage2.query(pairs), dim=1)
                    pooled = (pairs * w).sum(dim=1)
                    final = torch.cat([odd, pooled], dim=0)
                else:
                    pairs = out.view(-1, 2, out.size(-1))
                    w = torch.softmax(self.stage2.query(pairs), dim=1)
                    final = (pairs * w).sum(dim=1)
                stage2_list.append(final)
                final_lengths.append(final.shape[0])

            padded = pad_sequence(stage2_list, batch_first=True, padding_value=0.0, padding_side='left')
            return padded, torch.tensor(final_lengths, device=device)
    


    def forward(self, bert_profile_id, bert_profile_atten, bert_qa_id, bert_qa_atten,
                answer_start_indices, question_ids, question_attention_mask, labels,mode='original',num_finite=4):
        """
        前向传播函数，支持任意数量的 QA 对
        Forward pass of the model, supporting an arbitrary number of QA pairs
        """

        batch_size, qa_num, _ = bert_qa_id.shape

        # --- 1. BERT Encoding ---
        # 使用 BERT-style encoder-only model 编码输入文本 (history context)
        # Encode input text (history context) using BERT-style encoder-only model
        bert_forward_context = torch.no_grad() if self.freeze_bert else torch.enable_grad()
        with bert_forward_context:
            # 编码 profile
            # Encode profile
            bert_profile = self.bert_encoder(
                input_ids=bert_profile_id,
                attention_mask=bert_profile_atten,
                return_dict=True
            )
            # 编码所有 qa 对。为了并行处理，将 batch 和 qa_num 维度合并。
            # Encode all qa pairs. Merge batch and qa_num dimensions for parallel processing.
            bert_qa_outputs = self.bert_encoder(
                input_ids=bert_qa_id.view(-1, bert_qa_id.size(-1)),
                attention_mask=bert_qa_atten.view(-1, bert_qa_atten.size(-1)),
                return_dict=True
            )

        # 从 BERT 模型获取最终隐藏状态
        # Get the last hidden states from the BERT model
        encoding_profile = bert_profile.last_hidden_state           # [B, L_p, D_bert]
        encoding_qa = bert_qa_outputs.last_hidden_state.view(batch_size, qa_num, -1, self.bert_encoder.config.hidden_size) # [B, N_qa, L_q, D_bert]

        # 计算每个序列的实际长度（非填充部分）
        # Calculate actual lengths (non-padded parts) of each sequence
        len_profile = torch.sum(bert_profile_atten, dim=1).long()    # [B]
        len_qa = torch.sum(bert_qa_atten, dim=2).long()              # [B, N_qa]

        # 提取非填充部分（左填充时取最后若干个 token）
        # Extract non-padded parts (take last L tokens for left padding)
        non_padded_profile = [encoding_profile[i, -len_profile[i]:] for i in range(batch_size)]
        non_padded_qa = [[encoding_qa[i, j, -len_qa[i, j]:] for j in range(qa_num)] for i in range(batch_size)]

        # 将 profile + 所有 qa 的非填充编码拼接在一起
        # Concatenate non-padded encodings of profile + all qas
        encoding_list = []
        for i in range(batch_size):
            # 创建一个包含 profile 编码的列表
            # Create a list containing the profile encoding
            seq_list = [non_padded_profile[i]]
            # 遍历所有 qa 对并添加到列表中
            # Iterate through all qa pairs and add them to the list
            for j in range(qa_num):
                seq_list.append(non_padded_qa[i][j])
            # 拼接列表中的所有张量
            # Concatenate all tensors in the list
            encoding_list.append(torch.cat(seq_list, dim=0))

        # 将可变长度序列 pad 成固定长度张量
        # Pad variable-length sequences into fixed-length tensor
        padded_encoding = pad_sequence(encoding_list, batch_first=True,padding_value=0.0, padding_side="left")  # [B, max_L_total, D_bert]

        # --- 2. 构建答案相关的编码序列 (profile + 所有答案) ---
        # --- 2. Construct answer-related encoding (profile + all answers) ---
        answer_encodings_list = []
        answer_encodings_list_len = []
        for i in range(batch_size):
            # 创建一个包含 profile 编码的列表
            # Create a list containing the profile encoding
            seq_list = [non_padded_profile[i]]
            # 遍历所有 qa 对的答案部分并添加到列表中
            # Iterate through all qa pairs' answer spans and add them to the list
            for j in range(qa_num):
                # 获取答案起始位置
                # Get the answer start index
                start_qa = answer_start_indices[i, j].item()
                # 从非填充的编码中提取答案部分
                # Extract the answer span from the non-padded embedding
                answer_span = non_padded_qa[i][j][start_qa:]  # [L_aj, D_bert]
                seq_list.append(answer_span)
            
            # 拼接 profile 和所有答案
            # Concatenate profile and all answers
            concatenated_answer = torch.cat(seq_list, dim=0)
            answer_encodings_list.append(concatenated_answer)
            answer_encodings_list_len.append(len(concatenated_answer))

        # 将答案编码序列 pad 成统一长度
        # Pad answer encodings to same length
        padded_answer = pad_sequence(answer_encodings_list, batch_first=True,padding_value=0.0, padding_side="left")  # [B, max_L_a, D_bert]

        # --- 3. 两阶段注意力池化（Two-stage attention pooling）---
        # --- 3. Two-stage attention pooling ---
        max_seq_len_qa = padded_encoding.shape[1]
        max_seq_len_a = padded_answer.shape[1]

        # 计算每条样本的有效长度（真实 token 数）
        # Effective lengths (number of real tokens per sample)
        effective_lengths_qa = len_profile + torch.sum(len_qa, dim=1)  # [B]
        effective_lengths_a = torch.tensor(answer_encodings_list_len, dtype=torch.long, device=padded_answer.device)  # [B]
        
        x_qa, final_effective_lengths_qa = self._apply_two_stage_pooling(padded_encoding, effective_lengths_qa, max_seq_len_qa)
        x_a, final_effective_lengths_a = self._apply_two_stage_pooling(padded_answer, effective_lengths_a, max_seq_len_a)
        
        # --- 4. MLP + Codebook + MLP ---
        for name, param in self.codebook.named_parameters():
            _ = param

        user_model = self.codebook(x_a, final_effective_lengths_a, mode=mode, num_finite=num_finite) 

        # 构造最终输入序列：[QA pooled] + [User Model]
        # construct final input sequence: qa pooled + user model
        x_list = []
        x_list_len = torch.zeros(batch_size, dtype=torch.int64, device=padded_answer.device)

        # 根据不同的模式处理 user_model
        # deal with user_model based on mode
        if mode == "original":

            for i in range(batch_size):
                effective_len_qa = final_effective_lengths_qa[i]
                effective_len_a = final_effective_lengths_a[i]

                if effective_len_a > 0:
                    
                    user_seq = user_model[i, -effective_len_a:, :]  # [effective_len_a, D_codebook]
                else:
                    user_seq = user_model.new_empty(0, user_model.size(-1))

              
                qa_seq = x_qa[i, -effective_len_qa:]

                
                combined = torch.cat([qa_seq, user_seq], dim=0)
                x_list.append(combined)
                x_list_len[i] = len(combined)

        elif mode == "soft_prompt":
            
            for i in range(batch_size):
                effective_len_qa = final_effective_lengths_qa[i]
                
                
                user_vector = user_model[i].unsqueeze(0)  # [1, D_codebook]

                
                qa_seq = x_qa[i, -effective_len_qa:]

                
                combined = torch.cat([qa_seq, user_vector], dim=0)
                x_list.append(combined)
                x_list_len[i] = len(combined)

        elif mode == "finite":
          
            for i in range(batch_size):
                effective_len_qa = final_effective_lengths_qa[i]
                
                
                user_seq = user_model[i] # [num, D_codebook]

            
                qa_seq = x_qa[i, -effective_len_qa:]

            
                combined = torch.cat([qa_seq, user_seq], dim=0)
                x_list.append(combined)
                x_list_len[i] = len(combined)
            
    
        x = pad_sequence(x_list, batch_first=True, padding_value=0.0, padding_side="left")


        x = self.mlp1(x)

        for idx, layer in enumerate(self.mlp1):
            if isinstance(layer, nn.Linear):
                _ = layer.weight
                _ = layer.bias

        history_prompt_embeds = x.view(batch_size, -1, self.llama_dim)

        # --- 5. 获取问题嵌入 ---
        # --- 5. Get question embeddings ---
        decoding_emb = self.llama.get_input_embeddings()(question_ids)

        labels[labels == self.pad_token_id] = -100

        # --- 6. 构造 LLaMA 输入：在特定 token 处插入历史 prompt ---
        # --- 6. Construct LLaMA input: insert history prompt at special token ---
        match_mask = (question_ids == self.bert_token_id)
        if not match_mask.any(dim=1).all():
            raise ValueError("some sample do not have <BERT>")

        insert_pos = match_mask.float().argmax(dim=1)

        before_emb = []
        after_emb = []
        before_mask = []
        after_mask = []
        for b in range(batch_size):
            idx = insert_pos[b].item()
            before_emb.append(decoding_emb[b, :idx])
            after_emb.append(decoding_emb[b, idx + 1:])
            before_mask.append(question_attention_mask[b, :idx])
            after_mask.append(question_attention_mask[b, idx + 1:])

        valid_history_prompt_embeds = []
        valid_history_mask = []
        max_history_len = x_list_len.max().item()
        for i in range(batch_size):
            valid_len = x_list_len[i].item()
            start_idx = max_history_len - valid_len
            valid_history_prompt_embeds.append(history_prompt_embeds[i, start_idx:])
            valid_history_mask.append(torch.ones(valid_len, dtype=torch.int64, device=history_prompt_embeds.device))

        combined_embeddings_list = [
            torch.cat([before_emb[b], valid_history_prompt_embeds[b], after_emb[b]], dim=0)
            for b in range(batch_size)
        ]
        combined_masks_list = [
            torch.cat([before_mask[b], valid_history_mask[b], after_mask[b]], dim=0)
            for b in range(batch_size)
        ]

        inputs_embeds = pad_sequence(combined_embeddings_list, batch_first=True, padding_value=0.0, padding_side="left")
        final_attention_mask = pad_sequence(combined_masks_list, batch_first=True, padding_value=0, padding_side="left")

        if labels is not None:
            input_len = inputs_embeds.shape[1]
            answer_len = labels.shape[1]
            adjusted_labels = torch.full((batch_size, input_len), -100, dtype=labels.dtype, device=labels.device)
            adjusted_labels[:, -answer_len:] = labels
            final_labels = adjusted_labels
        else:
            final_labels = None

        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=final_attention_mask,
            labels=final_labels,
            return_dict=True
        )

        if outputs.loss is not None:
            total_loss = outputs.loss
            outputs.loss = total_loss

        return outputs

    @torch.no_grad()
    def generate_answer(self, bert_qa_id, bert_qa_atten,
                        answer_start_indices, question_ids, question_attention_mask, 
                        max_new_tokens=20, mode='original', num_finite=4):
        """
        Generates an answer using the multimodal model.
        使用多模态模型生成答案。

        Args:
            bert_profile_id (Tensor): Token IDs for the user profile.
                                    用户画像的 token ID。
            bert_profile_atten (Tensor): Attention mask for the user profile.
                                        用户画像的 attention mask。
            bert_qa_id (Tensor): Token IDs for multiple QA pairs.
                                多个问答对的 token ID。
            bert_qa_atten (Tensor): Attention masks for multiple QA pairs.
                                    多个问答对的 attention mask。
            answer_start_indices (Tensor): Starting indices of answers in each QA pair.
                                        每个问答对中答案的起始索引。
            question_ids (Tensor): Token IDs for the question prompt.
                                问题提示的 token ID。
            question_attention_mask (Tensor): Attention mask for the question prompt.
                                            问题提示的 attention mask。
            max_new_tokens (int): Maximum tokens to generate.
                                最大生成 token 数。
            mode (str): The operational mode for the codebook attention.
                        码本注意力的操作模式。
            num_finite (int): The number of groups for the 'finite' mode.
                            'finite' 模式下的分组数。
        
        Returns:
            Tensor: Generated token IDs.
                    生成的 token ID。
        """
        self.eval()
        batch_size, qa_num, _ = bert_qa_id.shape
        
        # --- 1. BERT Encoding ---
        # 使用 BERT-style encoder-only model 编码输入文本
        # Encode input text using BERT-style encoder-only model
        with torch.no_grad():
            
            # 编码所有 qa 对。为了并行处理，将 batch 和 qa_num 维度合并。
            # Encode all qa pairs. Merge batch and qa_num dimensions for parallel processing.
            bert_qa_outputs = self.bert_encoder(
                input_ids=bert_qa_id.view(-1, bert_qa_id.size(-1)),
                attention_mask=bert_qa_atten.view(-1, bert_qa_atten.size(-1)),
                return_dict=True
            )

        # 从 BERT 模型获取最终隐藏状态
        # Get the last hidden states from the BERT model
        # 将 QA 编码重新 reshape 回原始形状
        # Reshape QA embeddings back to original shape
        encoding_qa = bert_qa_outputs.last_hidden_state.view(batch_size, qa_num, -1, self.bert_encoder.config.hidden_size) # [B, N_qa, L_q, D_bert]

        # 计算每个序列的实际长度（非填充部分）
        # Calculate actual lengths (non-padded parts) of each sequence

        len_qa = torch.sum(bert_qa_atten, dim=2).long()            # [B, N_qa]

        # 提取非填充部分（左填充时取最后若干个 token）
        # Extract non-padded parts (take last L tokens for left padding)
        non_padded_qa = [[encoding_qa[i, j, -len_qa[i, j]:] for j in range(qa_num)] for i in range(batch_size)]

        # 将 profile + 所有 qa 的非填充编码拼接在一起
        # Concatenate non-padded encodings of profile + all qas
        encoding_list = []
        for i in range(batch_size):
            seq_list = []
            for j in range(qa_num):
                seq_list.append(non_padded_qa[i][j])
            encoding_list.append(torch.cat(seq_list, dim=0))

        padded_encoding = pad_sequence(encoding_list, batch_first=True, padding_value=0.0, padding_side="left")

        # --- 2. 构建答案相关的编码序列 (profile + 所有答案) ---
        # --- 2. Construct answer-related encoding (profile + all answers) ---
        answer_encodings_list = []
        answer_encodings_list_len = []
        for i in range(batch_size):
            seq_list = []
            for j in range(qa_num):
                start_qa = answer_start_indices[i, j].item()
                answer_span = non_padded_qa[i][j][start_qa:]
                seq_list.append(answer_span)
            
            concatenated_answer = torch.cat(seq_list, dim=0)
            answer_encodings_list.append(concatenated_answer)
            answer_encodings_list_len.append(len(concatenated_answer))

        padded_answer = pad_sequence(answer_encodings_list, batch_first=True, padding_value=0.0, padding_side="left")

        # --- 3. 两阶段注意力池化（Two-stage attention pooling）---
        # --- 3. Two-stage attention pooling ---
        max_seq_len_qa = padded_encoding.shape[1]
        max_seq_len_a = padded_answer.shape[1]

        effective_lengths_qa =  torch.sum(len_qa, dim=1)
        effective_lengths_a = torch.tensor(answer_encodings_list_len, dtype=torch.long, device=padded_answer.device)
        
        x_qa, final_effective_lengths_qa = self._apply_two_stage_pooling(padded_encoding, effective_lengths_qa, max_seq_len_qa)
        x_a, final_effective_lengths_a = self._apply_two_stage_pooling(padded_answer, effective_lengths_a, max_seq_len_a)

        # --- 4. MLP + Codebook + MLP ---
        # 使用 codebook 对用户特征进行建模，根据 mode 参数动态调整
        # Model user features using codebook, dynamically adjust based on mode
        user_model = self.codebook(x_a, final_effective_lengths_a, mode=mode, num_finite=num_finite) 

        # 构造最终输入序列：[QA pooled] + [User Model]
        # Construct final input: [QA pooled] + [User Model]
        x_list = []
        x_list_len = torch.zeros(batch_size, dtype=torch.int64, device=padded_answer.device)
        
        if mode == "original":
            for i in range(batch_size):
                effective_len_qa = final_effective_lengths_qa[i]
                effective_len_a = final_effective_lengths_a[i]

                if effective_len_a > 0:
                    user_seq = user_model[i, -effective_len_a:, :]
                else:
                    user_seq = user_model.new_empty(0, user_model.size(-1))

                qa_seq = x_qa[i, -effective_len_qa:]
                combined = torch.cat([qa_seq, user_seq], dim=0)
                x_list.append(combined)
                x_list_len[i] = len(combined)

        elif mode == "soft_prompt":
            for i in range(batch_size):
                effective_len_qa = final_effective_lengths_qa[i]
                user_vector = user_model[i].unsqueeze(0)

                qa_seq = x_qa[i, -effective_len_qa:]
                combined = torch.cat([qa_seq, user_vector], dim=0)
                x_list.append(combined)
                x_list_len[i] = len(combined)

        elif mode == "finite":
            for i in range(batch_size):
                effective_len_qa = final_effective_lengths_qa[i]
                user_seq = user_model[i]

                qa_seq = x_qa[i, -effective_len_qa:]
                combined = torch.cat([qa_seq, user_seq], dim=0)
                x_list.append(combined)
                x_list_len[i] = len(combined)
            
        x = pad_sequence(x_list, batch_first=True, padding_value=0.0, padding_side="left")
        x = self.mlp1(x)
        history_prompt_embeds = x.view(batch_size, -1, self.llama_dim)

        # --- 5. 获取问题嵌入 ---
        # --- 5. Get question embeddings ---
        decoding_emb = self.llama.get_input_embeddings()(question_ids)

        # --- 6. 构造 LLaMA 输入：在特定 token 处插入历史 prompt ---
        # --- 6. Construct LLaMA input: insert history prompt at special token ---
        match_mask = (question_ids == self.bert_token_id)
        if not match_mask.any(dim=1).all():
            raise ValueError("some sample do not have <BERT>")

        insert_pos = match_mask.float().argmax(dim=1)

        before_emb = []
        after_emb = []
        before_mask = []
        after_mask = []
        for b in range(batch_size):
            idx = insert_pos[b].item()
            before_emb.append(decoding_emb[b, :idx])
            after_emb.append(decoding_emb[b, idx + 1:])
            before_mask.append(question_attention_mask[b, :idx])
            after_mask.append(question_attention_mask[b, idx + 1:])

        valid_history_prompt_embeds = []
        valid_history_mask = []
        max_history_len = x_list_len.max().item()
        for i in range(batch_size):
            valid_len = x_list_len[i].item()
            start_idx = max_history_len - valid_len
            valid_history_prompt_embeds.append(history_prompt_embeds[i, start_idx:])
            valid_history_mask.append(torch.ones(valid_len, dtype=torch.int64, device=history_prompt_embeds.device))

        combined_embeddings_list = [
            torch.cat([before_emb[b], valid_history_prompt_embeds[b], after_emb[b]], dim=0)
            for b in range(batch_size)
        ]
        combined_masks_list = [
            torch.cat([before_mask[b], valid_history_mask[b], after_mask[b]], dim=0)
            for b in range(batch_size)
        ]

        inputs_embeds = pad_sequence(combined_embeddings_list, batch_first=True, padding_value=0.0, padding_side="left")
        final_attention_mask = pad_sequence(combined_masks_list, batch_first=True, padding_value=0, padding_side="left")

        # 7. 生成
        outputs = self.llama.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=final_attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id = self.pad_token_id,
            eos_token_id = 128009,
        )
        
        return outputs