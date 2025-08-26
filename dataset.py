import json
import os
from typing import Union

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer,  PreTrainedTokenizerFast
import torch
import warnings


class SSA_data(Dataset):
    """
    Dataset class for training a model with a BERT family encoder and a LLM decoder.
    该数据集类用于训练一个包含BERT类编码器和LLM解码器的模型.
    """
    def __init__(self, 
                 data_path: str, 
                 encoder: Union[PreTrainedTokenizer,  PreTrainedTokenizerFast], 
                 decoder: Union[PreTrainedTokenizer,  PreTrainedTokenizerFast], 
                 decoder_max_question_len: int = 150, 
                 decoder_max_answer_len: int = 128):
        """
        Initializes the SSA dataset.
        初始化SSA数据集.

        Args:
            data_path (str): The path to the data file (.json or .jsonl).
                             数据文件路径（.json或.jsonl）.
            encoder (Union[PreTrainedTokenizer,  PreTrainedTokenizerFast]): The tokenizer for the encoder (e.g.,  BERT).
                                                                            编码器的分词器（例如BERT）.
            decoder (Union[PreTrainedTokenizer,  PreTrainedTokenizerFast]): The tokenizer for the decoder (e.g.,  Qwen).
                                                                            解码器的分词器（例如Qwen）.
            decoder_max_question_len (int): The maximum length for the decoder's question input.
                                            解码器问题输入的最大长度.
            decoder_max_answer_len (int): The maximum length for the decoder's answer labels.
                                          解码器答案标签的最大长度.
        """
        self.data_path = data_path
        self.raw_data = self._load_data(data_path)
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_max_question_len = decoder_max_question_len
        self.decoder_max_answer_len = decoder_max_answer_len

        # Special token IDs for BERT
        # BERT的特殊token ID
        self.cls_token_id = encoder.cls_token_id
        self.sep_token_id = encoder.sep_token_id
        self.bert_pad_token_id = encoder.pad_token_id
        
        # Ensure the decoder tokenizer has a padding token.
        # 确保解码器分词器有填充token.
        if self.decoder.pad_token_id is None:
            warnings.warn("Decoder tokenizer lacks a pad token. Using EOS token as pad token.")
            self.decoder.pad_token = self.decoder.eos_token
            self.decoder.pad_token_id = self.decoder.eos_token_id

    def _load_data(self,  data_path: str) -> list:
        """
        Loads data from a file based on its extension.
        根据文件扩展名加载数据.

        Args:
            data_path (str): The path to the data file.
                             数据文件路径.

        Returns:
            list: A list of data samples.
                  数据样本列表.
        
        Raises:
            FileNotFoundError: If the data file does not exist.
                               如果data file does not esist.
            ValueError: If the file format is not supported.
                        如果文件格式不支持.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        ext = os.path.splitext(data_path)[1].lower()

        if ext == '.json':
            with open(data_path,  'r',  encoding='utf-8') as f:
                return json.load(f)
        elif ext == '.jsonl':
            data = []
            with open(data_path,  'r',  encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        else:
            raise ValueError(f"Unsupported file format: {ext}. Please use .json or .jsonl files.")

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        返回数据集中的样本数量.
        """
        return len(self.raw_data)

    def __getitem__(self,  idx):
        """
        Retrieves a single data sample and tokenizes it.
        获取单个数据样本并进行分词.

        Args:
            idx (int): The index of the item to retrieve.
                       要获取的项的索引.

        Returns:
            dict: A dictionary containing tokenized inputs and labels.
                  一个包含分词后的输入和标签的字典.
        """
        item = self.raw_data[idx]
        text = item.get("text",  "")

        # --- 1. Process for BERT Encoder ---
        # --- 1. 为BERT编码器进行处理 ---
        self.encoder.padding_side = "left"
        self.decoder.padding_side = "left"
        bert_encoded_inputs = self.encoder(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=128, 
            return_tensors="pt", 
            add_special_tokens=True
        )
        bert_input_ids = bert_encoded_inputs['input_ids'].squeeze(0)
        bert_attention_mask = bert_encoded_inputs['attention_mask'].squeeze(0)

        # --- 2. Process for Decoder (Question) ---
        # --- 2. 为解码器进行处理（问题） ---
        question_text = self._create_prompt(text)
        question_encoding = self.decoder(
            question_text, 
            max_length=self.decoder_max_question_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        question_ids = question_encoding['input_ids'].squeeze(0)
        question_attention_mask = question_encoding['attention_mask'].squeeze(0)

        # --- 3. Process for Decoder (Labels) ---
        # --- 3. 为解码器进行处理（标签） ---
        answer_text = text + self.decoder.eos_token
        labels_encoding = self.decoder(
            answer_text, 
            max_length=self.decoder_max_answer_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        labels = labels_encoding['input_ids'].squeeze(0)
        

        return {
            'bert_input_ids': bert_input_ids, 
            'bert_attention_mask': bert_attention_mask, 
            'question_ids': question_ids, 
            'question_attention_mask': question_attention_mask, 
            'labels': labels, 
        }

    def _create_prompt(self,  text: str) -> str:
        """
        Creates a formatted prompt using a messages-based structure.
        使用基于消息的结构创建格式化的提示.
        """
        messages = [
            {"role": "system",  "content": "You are a helpful assistant."}, 
            {"role": "user",  "content": f"Original Information: <BERT>\n\nYour task: Share a concise interpretation of the information provided:\n"}, 
            {"role":"assistant",  "content": text}
        ]
        return self.decoder.apply_chat_template(messages,  tokenize=False,  add_generation_prompt=False)


class SSA_infer(Dataset):
    """
    Dataset class for training a model with a BERT family encoder and a LLM decoder.
    该数据集类用于训练一个包含BERT类编码器和LLM解码器的模型.
    """
    def __init__(self, 
                 data_path: str, 
                 encoder: Union[PreTrainedTokenizer,  PreTrainedTokenizerFast], 
                 decoder: Union[PreTrainedTokenizer,  PreTrainedTokenizerFast], 
                 decoder_max_question_len: int = 150, 
                 decoder_max_answer_len: int = 128):
        """
        Initializes the SSA dataset.
        初始化SSA数据集.

        Args:
            data_path (str): The path to the data file (.json or .jsonl).
                             数据文件路径（.json或.jsonl）.
            encoder (Union[PreTrainedTokenizer,  PreTrainedTokenizerFast]): The tokenizer for the encoder (e.g.,  BERT).
                                                                            编码器的分词器（例如BERT）.
            decoder (Union[PreTrainedTokenizer,  PreTrainedTokenizerFast]): The tokenizer for the decoder (e.g.,  Qwen).
                                                                            解码器的分词器（例如Qwen）.
            decoder_max_question_len (int): The maximum length for the decoder's question input.
                                            解码器问题输入的最大长度.
            decoder_max_answer_len (int): The maximum length for the decoder's answer labels.
                                          解码器答案标签的最大长度.
        """
        self.data_path = data_path
        self.raw_data = self._load_data(data_path)
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_max_question_len = decoder_max_question_len
        self.decoder_max_answer_len = decoder_max_answer_len

        # Special token IDs for BERT
        # BERT的特殊token ID
        self.cls_token_id = encoder.cls_token_id
        self.sep_token_id = encoder.sep_token_id
        self.bert_pad_token_id = encoder.pad_token_id
        
        # Ensure the decoder tokenizer has a padding token.
        # 确保解码器分词器有填充token.
        if self.decoder.pad_token_id is None:
            warnings.warn("Qwen tokenizer lacks a pad token. Using EOS token as pad token.")
            self.decoder.pad_token = self.decoder.eos_token
            self.decoder.pad_token_id = self.decoder.eos_token_id

    def _load_data(self,  data_path: str) -> list:
        """
        Loads data from a file based on its extension.
        根据文件扩展名加载数据.

        Args:
            data_path (str): The path to the data file.
                             数据文件路径.

        Returns:
            list: A list of data samples.
                  数据样本列表.
        
        Raises:
            FileNotFoundError: If the data file does not exist.
                               如果data file does not esist.
            ValueError: If the file format is not supported.
                        如果文件格式不支持.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        ext = os.path.splitext(data_path)[1].lower()

        if ext == '.json':
            with open(data_path,  'r',  encoding='utf-8') as f:
                return json.load(f)
        elif ext == '.jsonl':
            data = []
            with open(data_path,  'r',  encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        else:
            raise ValueError(f"Unsupported file format: {ext}. Please use .json or .jsonl files.")

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        返回数据集中的样本数量.
        """
        return len(self.raw_data)

    def __getitem__(self,  idx):
        """
        Retrieves a single data sample and tokenizes it.
        获取单个数据样本并进行分词.

        Args:
            idx (int): The index of the item to retrieve.
                       要获取的项的索引.

        Returns:
            dict: A dictionary containing tokenized inputs and labels.
                  一个包含分词后的输入和标签的字典.
        """
        item = self.raw_data[idx]
        text = item.get("text",  "")

        # --- 1. Process for BERT Encoder ---
        # --- 1. 为BERT编码器进行处理 ---
        self.encoder.padding_side = "left"
        self.decoder.padding_side = "left"
        bert_encoded_inputs = self.encoder(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=128, 
            return_tensors="pt", 
            add_special_tokens=True
        )
        bert_input_ids = bert_encoded_inputs['input_ids'].squeeze(0)
        bert_attention_mask = bert_encoded_inputs['attention_mask'].squeeze(0)

        # --- 2. Process for Decoder (Question) ---
        # --- 2. 为解码器进行处理（问题） ---
        question_text = self._create_prompt(text)
        question_encoding = self.decoder(
            question_text, 
            max_length=self.decoder_max_question_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        question_ids = question_encoding['input_ids'].squeeze(0)
        question_attention_mask = question_encoding['attention_mask'].squeeze(0)

        # --- 3. Process for Decoder (Labels) ---
        # --- 3. 为解码器进行处理（标签） ---
        answer_text = text + self.decoder.eos_token
        labels_encoding = self.decoder(
            answer_text, 
            max_length=self.decoder_max_answer_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        labels = labels_encoding['input_ids'].squeeze(0)
        

        return {
            'bert_input_ids': bert_input_ids, 
            'bert_attention_mask': bert_attention_mask, 
            'question_ids': question_ids, 
            'question_attention_mask': question_attention_mask, 
            'labels': labels, 
        }

    def _create_prompt(self,  text: str) -> str:
        """
        Creates a formatted prompt using a messages-based structure.
        使用基于消息的结构创建格式化的提示.
        """
        messages = [
            {"role": "system",  "content": "You are a helpful assistant."}, 
            {"role": "user",  "content": f"Original Information: <BERT>\n\nYour task: Share a concise interpretation of the information provided:\n"}, 
        ]
        return self.decoder.apply_chat_template(messages,  tokenize=False,  add_generation_prompt=True)





def simple_collate_fn(batch):
    """
    A simple collate function to stack tensors from a batch of samples.
    一个简单的collate函数, 用于堆叠批量样本中的张量.
    """
    elem = batch[0]
    batch_dict = {key: [d[key] for d in batch] for key in elem}
    return {key: torch.stack(values) for key,  values in batch_dict.items()}





class UBC_data(Dataset):
    """
    Dataset class for training a multimodal model with BERT-based encoders and an LLM decoder (e.g.,  LLaMA).
    用于训练包含BERT类编码器和LLM解码器（例如LLaMA）的多模态模型的数据集类.
    
    This dataset processes user profiles and related Q&A pairs separately using BERT,  and constructs 
    instruction-tuned prompts for the LLM decoder. It supports dynamic prompt templating via chat templates.
    该数据集使用BERT分别处理用户画像和相关问答对, 并为LLM解码器构建指令微调提示.支持通过对话模板动态构造提示.
    """
    def __init__(self, 
                 data_path: str, 
                 bert_tokenizer: Union[PreTrainedTokenizer,  PreTrainedTokenizerFast], 
                 llama_tokenizer: Union[PreTrainedTokenizer,  PreTrainedTokenizerFast], 
                 bert_max_len: int = 400, 
                 llama_max_question_len: int = 256, 
                 llama_max_answer_len: int = 256, 
                 qa_num: int = 2):  
        """
        Initializes the UBC dataset.
        初始化UBC数据集.

        Args:
            data_path (str): The path to the data file (.json or .jsonl).
                             数据文件路径（.json或.jsonl）.
            bert_tokenizer (Union[PreTrainedTokenizer,  PreTrainedTokenizerFast]): Tokenizer for BERT encoder.
                                                                                 BERT编码器的分词器.
            llama_tokenizer (Union[PreTrainedTokenizer,  PreTrainedTokenizerFast]): Tokenizer for LLaMA decoder.
                                                                                 LLaMA解码器的分词器.
            bert_max_len (int): Maximum sequence length for BERT inputs (for QA entries).
                                BERT输入的最大序列长度（用于问答条目）.
            llama_max_question_len (int): Maximum length for the decoder's input prompt.
                                          解码器输入提示的最大长度.
            llama_max_answer_len (int): Maximum length for the decoder's answer labels.
                                        解码器答案标签的最大长度.
            qa_num (int): The number of QA pairs to process for each sample.
                          每个样本要处理的问答对数量.
        """
        self.data_path = data_path
        self.raw_data = self._load_data(data_path)

        self.bert_tokenizer = bert_tokenizer
        self.llama_tokenizer = llama_tokenizer
        self.bert_max_len = bert_max_len
        self.llama_max_question_len = llama_max_question_len
        self.llama_max_answer_len = llama_max_answer_len
        self.qa_num = qa_num 

        self.bert_pad_token_id = self.bert_tokenizer.pad_token_id

        # Ensure the LLaMA tokenizer has a padding token.
        # 确保LLaMA分词器具有填充token.
        if self.llama_tokenizer.pad_token_id is None:
            print("警告: LLaMA tokenizer 缺少 pad token, 将使用 EOS token 作为 pad token. Warn: LLaMA tokenizer lacks a pad token. Using EOS token as pad token.")
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            self.llama_tokenizer.pad_token_id = self.llama_tokenizer.eos_token_id

    def _load_data(self,  data_path: str) -> list:
        """
        Loads data from a file based on its extension.
        根据文件扩展名加载数据.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file does not exist: {data_path}")

        ext = os.path.splitext(data_path)[1].lower()

        if ext == '.json':
            with open(data_path,  'r',  encoding='utf-8') as f:
                return json.load(f)
        elif ext == '.jsonl':
            data = []
            with open(data_path,  'r',  encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        else:
            raise ValueError(f"Do not support this file format: {ext}, please use .json or .jsonl ")

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        返回数据集中的样本数量.
        """
        return len(self.raw_data)

    def __getitem__(self,  idx):
        """
        Retrieves a single data sample and tokenizes it for both BERT encoders and LLaMA decoder.
        获取单个数据样本, 并为BERT编码器和LLaMA解码器进行分词处理.
        """
        item = self.raw_data[idx]
        self.bert_tokenizer.padding_side = "left"
        self.llama_tokenizer.padding_side = "left"
        
        # --- 1. Encode User Profile with BERT ---
        # --- 1. 使用BERT编码用户画像 ---
        profile = item.get("profile",  "")
        profile_encoding = self.bert_tokenizer(
            "The user's profile: " + profile, 
            max_length=100, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt"
        )
        bert_profile_id = profile_encoding['input_ids'].squeeze(0)
        bert_profile_atten = profile_encoding['attention_mask'].squeeze(0)

        # --- 2. Process and Encode Related Q&A Pairs ---
        # --- 2. 处理并编码相关问答对 ---
        # 根据 qa_num 动态选择和处理 QA 对
        # Dynamically choose and select QA pairs via qa_num
        related_qa_list = sorted(item.get("related Q&A",  []),  key=lambda x: len(x))[:self.qa_num]
        
        qa_input_ids_list = []
        qa_attention_masks_list = []
        answer_start_indices_list = []

        for qa in related_qa_list:
            qa = qa.replace("Q:",  "Question:").replace("A:",  "Answer:")
            input_text = "The user's history Q&A includes: " + qa

            encoding = self.bert_tokenizer(
                input_text, 
                max_length=self.bert_max_len, 
                padding='max_length', 
                truncation=True, 
                return_tensors="pt", 
                add_special_tokens=True
            )

            qa_input_ids = encoding['input_ids'].squeeze(0)
            qa_attention_mask = encoding['attention_mask'].squeeze(0)

            # Find the starting token position of "Answer:"
            tokens = self.bert_tokenizer.convert_ids_to_tokens(qa_input_ids)
            target_subtokens = self.bert_tokenizer.tokenize("Answer:")
            target_len = len(target_subtokens)

            answer_start_idx = -1
            for i in range(len(tokens) - target_len + 1):
                if tokens[i:i + target_len] == target_subtokens:
                    answer_start_idx = i + target_len
                    break

            qa_input_ids_list.append(qa_input_ids)
            qa_attention_masks_list.append(qa_attention_mask)
            answer_start_indices_list.append(answer_start_idx)

        # Pad to ensure exactly qa_num entries
        while len(qa_input_ids_list) < self.qa_num:
            pad_input = torch.full((self.bert_max_len, ),  self.bert_pad_token_id,  dtype=torch.long)
            pad_mask = torch.zeros(self.bert_max_len,  dtype=torch.long)
            qa_input_ids_list.append(pad_input)
            qa_attention_masks_list.append(pad_mask)
            answer_start_indices_list.append(-1)
        
        # Stack the lists into single tensors
        bert_qa_id = torch.stack(qa_input_ids_list)
        bert_qa_atten = torch.stack(qa_attention_masks_list)
        answer_start_indices = torch.tensor(answer_start_indices_list,  dtype=torch.long)

        # --- 3. Construct Prompt for LLaMA Decoder ---
        # --- 3. 为LLaMA解码器构建提示 ---
        question_text = item.get("Question",  "")
        answer_text = item.get("answer",  "")

        chat = [
            {
                "role": "system", 
                "content": (
                    "You are an intelligent assistant skilled at mimicking a user's behavior"
                    "based on their social media profile and history interaction with a user model."
                )
            }, 
            {
                "role": "user", 
                "content": f"The user's history interaction and user model are:\n\n <BERT>. Now,  please answer the following question as if you were this user:\n {question_text}"
            }, 
            {
                "role": "assistant", 
                "content": f"{answer_text}"
            }
        ]
        full_prompt = self.llama_tokenizer.apply_chat_template(chat,  tokenize=False,  add_generation_prompt=False)
        question_encoding = self.llama_tokenizer(
            full_prompt, 
            max_length=self.llama_max_question_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        question_ids = question_encoding['input_ids'].squeeze(0)
        question_attention_mask = question_encoding['attention_mask'].squeeze(0)

        # --- 4. Tokenize Answer as Labels ---
        # --- 4. 将答案作为标签进行分词 ---
        answer_encoding = self.llama_tokenizer(
            answer_text.strip() + self.llama_tokenizer.eos_token, 
            max_length=self.llama_max_answer_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        labels = answer_encoding['input_ids'].squeeze(0)

        return {
            'bert_profile_id': bert_profile_id, 
            'bert_profile_atten': bert_profile_atten, 
            'bert_qa_id': bert_qa_id, 
            'bert_qa_atten': bert_qa_atten, 
            'answer_start_indices': answer_start_indices, 
            'question_ids': question_ids, 
            'question_attention_mask': question_attention_mask, 
            'labels': labels
        }



class UBC_infer(Dataset):
    """
    Dataset class for inference with a multimodal model using BERT encoders and an LLM decoder (e.g.,  LLaMA).
    用于推理阶段的多模态模型数据集类, 支持BERT编码器和LLM解码器（例如LLaMA）.

    This dataset processes user profiles and related Q&A pairs separately using BERT,  and constructs 
    instruction-tuned prompts for the LLM decoder. Designed for inference,  it excludes ground-truth 
    answer from the prompt and supports generation via `add_generation_prompt=True`.
    该数据集使用BERT分别处理用户画像和相关问答对, 并为LLM解码器构建指令提示.专为推理设计, 
    提示中不包含真实答案, 并通过 `add_generation_prompt=True` 支持自动生成.
    """
    def __init__(self, 
                 data_path: str, 
                 bert_tokenizer: Union[PreTrainedTokenizer,  PreTrainedTokenizerFast], 
                 llama_tokenizer: Union[PreTrainedTokenizer,  PreTrainedTokenizerFast], 
                 bert_max_len: int = 400, 
                 llama_max_question_len: int = 256, 
                 llama_max_answer_len: int = 256, 
                 qa_num: int = 2):
        """
        Initializes the UBC_infer dataset for inference tasks.
        初始化用于推理任务的UBC_infer数据集.

        Args:
            data_path (str): The path to the data file (.json or .jsonl).
                             数据文件路径（.json或.jsonl）.
            bert_tokenizer (Union[PreTrainedTokenizer,  PreTrainedTokenizerFast]): Tokenizer for BERT encoder.
                                                                                    BERT编码器的分词器.
            llama_tokenizer (Union[PreTrainedTokenizer,  PreTrainedTokenizerFast]): Tokenizer for LLaMA decoder.
                                                                                   LLaMA解码器的分词器.
            bert_max_len (int): Maximum sequence length for BERT inputs (for QA entries).
                                BERT输入的最大序列长度（用于问答条目）.
            llama_max_question_len (int): Maximum length for the decoder's input prompt.
                                          解码器输入提示的最大长度.
            llama_max_answer_len (int): Maximum length for the decoder's answer labels (used for label padding).
                                        解码器答案标签的最大长度（用于标签填充）.
        """
        self.data_path = data_path
        self.raw_data = self._load_data(data_path)
        self.raw_data = self.raw_data  # Limit data size for fast inference debugging
                                      # 限制数据量以加快推理调试速度

        self.bert_tokenizer = bert_tokenizer
        self.llama_tokenizer = llama_tokenizer
        self.bert_max_len = bert_max_len
        self.llama_max_question_len = llama_max_question_len
        self.llama_max_answer_len = llama_max_answer_len

        self.bert_pad_token_id = self.bert_tokenizer.pad_token_id
        self.qa_num=qa_num
        # Ensure the LLaMA tokenizer has a padding token.
        # 确保LLaMA分词器具有填充token.
        if self.llama_tokenizer.pad_token_id is None:
            print("警告: LLaMA tokenizer 缺少 pad token, 将使用 EOS token 作为 pad token.")
            warnings.warn("LLaMA tokenizer lacks a pad token. Using EOS token as pad token.")
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            self.llama_tokenizer.pad_token_id = self.llama_tokenizer.eos_token_id

    def _load_data(self,  data_path: str) -> list:
        """
        Loads data from a file based on its extension.
        根据文件扩展名加载数据.

        Args:
            data_path (str): The path to the data file.
                             数据文件路径.

        Returns:
            list: A list of data samples.
                  数据样本列表.

        Raises:
            FileNotFoundError: If the data file does not exist.
                               如果data file does not esist.
            ValueError: If the file format is not supported.
                        如果文件格式不支持.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file does not exist: {data_path}")

        ext = os.path.splitext(data_path)[1].lower()

        if ext == '.json':
            with open(data_path,  'r',  encoding='utf-8') as f:
                return json.load(f)
        elif ext == '.jsonl':
            data = []
            with open(data_path,  'r',  encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        else:
            raise ValueError(f"Do not support this file format: {ext}, please use .json or .jsonl file")

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        返回数据集中的样本数量.
        """
        return len(self.raw_data)

    def __getitem__(self,  idx):
        """
        Retrieves a single data sample and tokenizes it for both BERT encoders and LLaMA decoder.
        获取单个数据样本, 并为BERT编码器和LLaMA解码器进行分词处理.
        """
        item = self.raw_data[idx]
        self.bert_tokenizer.padding_side = "left"
        self.llama_tokenizer.padding_side = "left"
        
        # --- 1. Encode User Profile with BERT ---
        # --- 1. 使用BERT编码用户画像 ---
        profile = item.get("profile",  "")
        profile_encoding = self.bert_tokenizer(
            "The user's profile: " + profile, 
            max_length=100, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt"
        )
        bert_profile_id = profile_encoding['input_ids'].squeeze(0)
        bert_profile_atten = profile_encoding['attention_mask'].squeeze(0)

        # --- 2. Process and Encode Related Q&A Pairs ---
        # --- 2. 处理并编码相关问答对 ---
        # 根据 qa_num 动态选择和处理 QA 对
        # Dynamically choose and select QA pairs via qa_num
        related_qa_list = sorted(item.get("related Q&A",  []),  key=lambda x: len(x))[:self.qa_num]
        
        qa_input_ids_list = []
        qa_attention_masks_list = []
        answer_start_indices_list = []

        for qa in related_qa_list:
            qa = qa.replace("Q:",  "Question:").replace("A:",  "Answer:")
            input_text = "The user's history Q&A includes: " + qa

            encoding = self.bert_tokenizer(
                input_text, 
                max_length=self.bert_max_len, 
                padding='max_length', 
                truncation=True, 
                return_tensors="pt", 
                add_special_tokens=True
            )

            qa_input_ids = encoding['input_ids'].squeeze(0)
            qa_attention_mask = encoding['attention_mask'].squeeze(0)

            # Find the starting token position of "Answer:"
            tokens = self.bert_tokenizer.convert_ids_to_tokens(qa_input_ids)
            target_subtokens = self.bert_tokenizer.tokenize("Answer:")
            target_len = len(target_subtokens)

            answer_start_idx = -1
            for i in range(len(tokens) - target_len + 1):
                if tokens[i:i + target_len] == target_subtokens:
                    answer_start_idx = i + target_len
                    break

            qa_input_ids_list.append(qa_input_ids)
            qa_attention_masks_list.append(qa_attention_mask)
            answer_start_indices_list.append(answer_start_idx)

        # Pad to ensure exactly qa_num entries
        while len(qa_input_ids_list) < self.qa_num:
            pad_input = torch.full((self.bert_max_len, ),  self.bert_pad_token_id,  dtype=torch.long)
            pad_mask = torch.zeros(self.bert_max_len,  dtype=torch.long)
            qa_input_ids_list.append(pad_input)
            qa_attention_masks_list.append(pad_mask)
            answer_start_indices_list.append(-1)
        
        # Stack the lists into single tensors
        bert_qa_id = torch.stack(qa_input_ids_list)
        bert_qa_atten = torch.stack(qa_attention_masks_list)
        answer_start_indices = torch.tensor(answer_start_indices_list,  dtype=torch.long)

        # --- 3. Construct Prompt for LLaMA Decoder ---
        # --- 3. 为LLaMA解码器构建提示 ---
        question_text = item.get("Question",  "")
        answer_text = item.get("answer",  "")

        chat = [
            {
                "role": "system", 
                "content": (
                    "You are an intelligent assistant skilled at mimicking a user's behavior"
                    "based on their social media profile and history interaction with a user model."
                )
            }, 
            {
                "role": "user", 
                "content": f"The user's history interaction and user model are:\n\n <BERT>. Now,  please answer the following question as if you were this user:\n {question_text}\nWrite at least 40 words."
            }, 
        ]
        full_prompt = self.llama_tokenizer.apply_chat_template(
            chat,  
            tokenize=False,  
            add_generation_prompt=True  # Ensures proper formatting for text generation
                                      # 确保为文本生成提供正确格式
        )
        question_encoding = self.llama_tokenizer(
            full_prompt, 
            max_length=self.llama_max_question_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        question_ids = question_encoding['input_ids'].squeeze(0)
        question_attention_mask = question_encoding['attention_mask'].squeeze(0)

        # --- 4. Tokenize Answer as Labels (for Evaluation) ---
        # --- 4. 将答案作为标签进行分词（用于评估） ---
        answer_encoding = self.llama_tokenizer(
            answer_text.strip() + self.llama_tokenizer.eos_token, 
            max_length=128, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        labels = answer_encoding['input_ids'].squeeze(0)

        return {
            'bert_profile_id': bert_profile_id, 
            'bert_profile_atten': bert_profile_atten, 
            'bert_qa_id': bert_qa_id, 
            'bert_qa_atten': bert_qa_atten, 
            'answer_start_indices': torch.tensor(answer_start_indices,  dtype=torch.long), 
            'question_ids': question_ids, 
            'question_attention_mask': question_attention_mask, 
            'labels': labels
        }
    






class CSG_news(Dataset):
    """
    Dataset class for training a multimodal model with BERT-based encoders and an LLM decoder (e.g.,  LLaMA).
    用于训练包含BERT类编码器和LLM解码器（例如LLaMA）的多模态模型的数据集类.

    - 使用 BERT 对多个 profile（由 num_qa 控制）分别编码, 并返回 [num_qa,  L] 的张量.
    - 通过 chat template 构造 LLaMA 的输入, 并单独编码 labels.
    - 额外返回 answer_start_indices：记录每个 profile 中 "title:" 的 token 起始位置.
    """
    def __init__(
        self, 
        raw_path: str, 
        bert_tokenizer: Union[PreTrainedTokenizer,  PreTrainedTokenizerFast], 
        llama_tokenizer: Union[PreTrainedTokenizer,  PreTrainedTokenizerFast], 
        bert_max_len: int = 80, 
        llama_max_len: int = 128, 
        llama_max_answer_len: int = 16, 
        num_qa: int = 2
    ):
        self.raw_path = raw_path
        self.bert_tokenizer = bert_tokenizer
        self.llama_tokenizer = llama_tokenizer
        self.bert_max_len = bert_max_len
        self.llama_max_len = llama_max_len
        self.llama_max_answer_len = llama_max_answer_len
        self.num_qa = num_qa

        self.bert_pad_token_id = self.bert_tokenizer.pad_token_id

        # 确保 LLaMA tokenizer 有 pad token
        # Make sure LLaMA tokenizer has pad token
        if self.llama_tokenizer.pad_token_id is None:
            print("Warning: LLaMA tokenizer lacks pad token, will use EOS as pad token.")
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            self.llama_tokenizer.pad_token_id = self.llama_tokenizer.eos_token_id

        # 读数据
        # Read data
        self.raw_data = self._load_data(raw_path)


        # 组装样本
        # Deal with data
        self.samples = []
        for item in self.raw_data:
            sid = item["id"]

            # 取前 num_qa 个 profile, 不足补空
            # take the first num_qa profile
            profiles = item.get("profile",  [])[: self.num_qa]
            profile_prompts = [
                f"News Content: {p.get('text',  '')}\n\nNews title: {p.get('title',  '')}"
                for p in profiles
            ]
            if len(profile_prompts) < self.num_qa:
                profile_prompts += [""] * (self.num_qa - len(profile_prompts))

            input_prompt = (
                "Historical information and user model is:\n\n"
                "<BERT>\n\n"
                "Based on the user's historical summaries and title preferences provided above,  "
                "please complete the following instruction.\n\n"
                + item["input"]
                + "\nOnly output your headline alone.\n Write about 10 words.\n"
            )
            output_text = item["output"]

            self.samples.append({
                "profile_prompts": profile_prompts,   # list[str] 长度(length) == num_qa
                "input_prompt": input_prompt, 
                "output_text": output_text
            })

    def _load_data(self,  path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"data file does not esist: {path}")
        with open(path,  "r",  encoding="utf-8") as f:
            return json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,  idx):
        item = self.samples[idx]
        self.bert_tokenizer.padding_side = "left"
        self.llama_tokenizer.padding_side = "left"

        profile_prompts = item["profile_prompts"]
        input_prompt = item["input_prompt"]
        output_text = item["output_text"]

        # ===== 1) BERT 对 profile（当作 QA 条目）逐条编码, 并记录 "title:" 的 token 起始位置,后续构建user model =====
        # BERT encode profile and record the start position of title, which will be used to construct user model

        qa_input_ids_list = []
        qa_attention_masks_list = []
        title_start_indices_list = []

        # 预先准备候选子词序列, 增强健壮性（兼容 title:/Title:）
        # prepare target token
        target_variants = ["title:",  "Title:"]
        target_subtokens_variants = [self.bert_tokenizer.tokenize(tv) for tv in target_variants]

        for prof_text in profile_prompts:
            enc = self.bert_tokenizer(
                prof_text, 
                max_length=self.bert_max_len, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt", 
                add_special_tokens=True
            )
            ids = enc["input_ids"].squeeze(0)
            att = enc["attention_mask"].squeeze(0)

            # 在 token 序列中查找 "title:" 的起始 token 位置
            # find title: token start idx in token sequence
            tokens = self.bert_tokenizer.convert_ids_to_tokens(ids)
            found_idx = -1
            for target_subtokens in target_subtokens_variants:
                if not target_subtokens:
                    continue
                tlen = len(target_subtokens)
                for i in range(len(tokens) - tlen + 1):
                    if tokens[i:i + tlen] == target_subtokens:
                        found_idx = i +tlen  
                        break
                if found_idx != -1:
                    break

            qa_input_ids_list.append(ids)
            qa_attention_masks_list.append(att)
            title_start_indices_list.append(found_idx)

        # 堆叠成 [num_qa,  L]
        #stack into [num_qa,  L]
        bert_qa_id = torch.stack(qa_input_ids_list)
        bert_qa_atten = torch.stack(qa_attention_masks_list)
        answer_start_indices = torch.tensor(title_start_indices_list,  dtype=torch.long)

        # ===== 2) 构造 LLaMA 的 chat prompt（system + user + assistant）, 并编码 =====
        # construct LLaMA chat prompt and encode
        chat = [
            {"role": "system",  "content": "You are a helpful assistant good at drawing headline based on a given text."}, 
            {"role": "user",  "content": input_prompt}, 
        ]
        full_prompt = self.llama_tokenizer.apply_chat_template(
            chat,  tokenize=False,  add_generation_prompt=True
        )

        question_enc = self.llama_tokenizer(
            full_prompt, 
            max_length=self.llama_max_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        question_ids = question_enc["input_ids"].squeeze(0)
        question_attention_mask = question_enc["attention_mask"].squeeze(0)

        # ===== 3) labels：只编码答案文本 =====
        #labels: only encode answer text
        label_enc = self.llama_tokenizer(
            output_text.strip() + self.llama_tokenizer.eos_token, 
            max_length=self.llama_max_answer_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        labels = label_enc["input_ids"].squeeze(0)

        return {
            # BERT 编码（与 UBC_data 对齐的命名） BERT encode
            "bert_qa_id": bert_qa_id,                      # [num_qa,  L_bert]
            "bert_qa_atten": bert_qa_atten,                # [num_qa,  L_bert]
            "answer_start_indices": answer_start_indices,  # [num_qa] —— "title:" 的 token 起始位置（找不到为 -1） start idx of title: , if not found, -1
            # LLaMA 编码（与 UBC_data 对齐的命名） LLaMA encode
            "question_ids": question_ids, 
            "question_attention_mask": question_attention_mask, 
            "labels": labels
        }


class CSG_tweet(Dataset):
    """
    Dataset class for training a multimodal model with BERT-based encoders and an LLM decoder (e.g.,  LLaMA).
    用于训练包含BERT类编码器和LLM解码器（例如LLaMA）的多模态模型的数据集类.

    - 使用 BERT 对多个 profile（由 num_qa 控制）分别编码, 并返回 [num_qa,  L] 的张量.
    - 通过 chat template 构造 LLaMA 的输入, 并单独编码 labels.
    - 额外返回 answer_start_indices：记录每个 profile 中 "title:" 的 token 起始位置.
    """
    def __init__(
        self, 
        raw_path: str, 

        bert_tokenizer: Union[PreTrainedTokenizer,  PreTrainedTokenizerFast], 
        llama_tokenizer: Union[PreTrainedTokenizer,  PreTrainedTokenizerFast], 
        bert_max_len: int = 80, 
        llama_max_len: int = 128, 
        llama_max_answer_len: int = 16, 
        num_qa: int = 2
    ):
        self.raw_path = raw_path

        self.bert_tokenizer = bert_tokenizer
        self.llama_tokenizer = llama_tokenizer
        self.bert_max_len = bert_max_len
        self.llama_max_len = llama_max_len
        self.llama_max_answer_len = llama_max_answer_len
        self.num_qa = num_qa

        self.bert_pad_token_id = self.bert_tokenizer.pad_token_id

        # 确保 LLaMA tokenizer 有 pad token 
        # make sure LLaMA tokenizer has pad token
        if self.llama_tokenizer.pad_token_id is None:
            print("Warning: LLaMA tokenizer lacks pad token, will use EOS as pad token.")
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            self.llama_tokenizer.pad_token_id = self.llama_tokenizer.eos_token_id

        # 读数据 read data
        self.raw_data = self._load_data(raw_path)



        # 组装样本 construct sample
        self.samples = []
        for item in self.raw_data:
            sid = item["id"]


            # 取前 num_qa 个 profile, 不足补空  take first num_qa profile
            profiles = item.get("profile",  [])[: self.num_qa]
            profile_prompts = [
                f"Previous Tweets:\n{p['text']}\n\n"
                for p in profiles
            ]
            if len(profile_prompts) < self.num_qa:
                profile_prompts += [""] * (self.num_qa - len(profile_prompts))

            input_prompt = (
                "Previous tweets and user model is:\n\n"
                "<BERT>\n\n"
                "Based on the user's previous tweet and user model provided above,  "
                "please complete the following instruction.\n\n"
                + item["input"]
                + "\nOnly output your paraphrased whole tweet as if you are the user."
            )
            output_text = item["output"]

            self.samples.append({
                "profile_prompts": profile_prompts,   # len(list[str]) == num_qa
                "input_prompt": input_prompt, 
                "output_text": output_text
            })

    def _load_data(self,  path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"data file does not esist: {path}")
        with open(path,  "r",  encoding="utf-8") as f:
            return json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,  idx):
        item = self.samples[idx]
        self.bert_tokenizer.padding_side = "left"
        self.llama_tokenizer.padding_side = "left"

        profile_prompts = item["profile_prompts"]
        input_prompt = item["input_prompt"]
        output_text = item["output_text"]

        # ===== 1) BERT 对 profile（当作 QA 条目）逐条编码, 并记录 "tweet:" 的 token 起始位置 =====
        # BERT encode profile and record the start position of tweet, which will be used to construct user model
        qa_input_ids_list = []
        qa_attention_masks_list = []
        title_start_indices_list = []

        # 预先准备候选子词序列, 增强健壮性
        # prepare target words
        target_variants = ["Tweets:\n"]
        target_subtokens_variants = [self.bert_tokenizer.tokenize(tv) for tv in target_variants]

        for prof_text in profile_prompts:
            enc = self.bert_tokenizer(
                prof_text, 
                max_length=self.bert_max_len, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt", 
                add_special_tokens=True
            )
            ids = enc["input_ids"].squeeze(0)
            att = enc["attention_mask"].squeeze(0)

            # 在 token 序列中查找 "tweet:" 的起始 token 位置
            # find start token idx of tweets in sequense 
            tokens = self.bert_tokenizer.convert_ids_to_tokens(ids)
            found_idx = -1
            for target_subtokens in target_subtokens_variants:
                if not target_subtokens:
                    continue
                tlen = len(target_subtokens)
                for i in range(len(tokens) - tlen + 1):
                    if tokens[i:i + tlen] == target_subtokens:
                        found_idx = i +tlen 
                        break
                if found_idx != -1:
                    break

            qa_input_ids_list.append(ids)
            qa_attention_masks_list.append(att)
            title_start_indices_list.append(found_idx)

        # 堆叠成 [num_qa,  L] stack
        bert_qa_id = torch.stack(qa_input_ids_list)
        bert_qa_atten = torch.stack(qa_attention_masks_list)
        answer_start_indices = torch.tensor(title_start_indices_list,  dtype=torch.long)

        # ===== 2) 构造 LLaMA 的 chat prompt（system + user + assistant）, 并编码 =====
        # build LLaMA chat prompt
        chat = [
            {"role": "system",  "content": "You are a helpful assistant good at paraphrasing tweet based on previous tweets."}, 
            {"role": "user",  "content": input_prompt}, 
        ]
        full_prompt = self.llama_tokenizer.apply_chat_template(
            chat,  tokenize=False,  add_generation_prompt=True
        )

        question_enc = self.llama_tokenizer(
            full_prompt, 
            max_length=self.llama_max_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        question_ids = question_enc["input_ids"].squeeze(0)
        question_attention_mask = question_enc["attention_mask"].squeeze(0)

        # ===== 3) labels：只编码答案文本 =====
        # labels : only encode answer text
        label_enc = self.llama_tokenizer(
            output_text.strip() + self.llama_tokenizer.eos_token, 
            max_length=self.llama_max_answer_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        labels = label_enc["input_ids"].squeeze(0)

        return {
            # BERT 编码（与 UBC_data 对齐的命名）   bert encode
            "bert_qa_id": bert_qa_id,                      # [num_qa,  L_bert]
            "bert_qa_atten": bert_qa_atten,                # [num_qa,  L_bert]
            "answer_start_indices": answer_start_indices,  # [num_qa] —— "tweet"  token start idx（not found -1）
            # LLaMA 编码（与 UBC_data 对齐的命名）  llama encode
            "question_ids": question_ids, 
            "question_attention_mask": question_attention_mask, 
            "labels": labels
        }

class CSG_review(Dataset):
    """
    Dataset class for training a multimodal model with BERT-based encoders and an LLM decoder (e.g.,  LLaMA).
    用于训练包含BERT类编码器和LLM解码器（例如LLaMA）的多模态模型的数据集类.

    - 使用 BERT 对多个 profile（由 num_qa 控制）分别编码, 并返回 [num_qa,  L] 的张量.
    - 通过 chat template 构造 LLaMA 的输入, 并单独编码 labels.
    - 额外返回 answer_start_indices：记录每个 profile 中 "title:" 的 token 起始位置.
    """
    def __init__(
        self, 
        raw_path: str, 
        bert_tokenizer: Union[PreTrainedTokenizer,  PreTrainedTokenizerFast], 
        llama_tokenizer: Union[PreTrainedTokenizer,  PreTrainedTokenizerFast], 
        bert_max_len: int = 300, 
        llama_max_len: int = 200, 
        llama_max_answer_len: int = 16, 
        num_qa: int = 1
    ):
        self.raw_path = raw_path

        self.bert_tokenizer = bert_tokenizer
        self.llama_tokenizer = llama_tokenizer
        self.bert_max_len = bert_max_len
        self.llama_max_len = llama_max_len
        self.llama_max_answer_len = llama_max_answer_len
        self.num_qa = num_qa

        self.bert_pad_token_id = self.bert_tokenizer.pad_token_id

        # 确保 LLaMA tokenizer 有 pad token 
        # make sure LLaMA tokenizer has pad token
        if self.llama_tokenizer.pad_token_id is None:
            print("Warning: LLaMA tokenizer lacks pad token, will use EOS as pad token.")
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            self.llama_tokenizer.pad_token_id = self.llama_tokenizer.eos_token_id

        
        self.raw_data = self._load_data(raw_path)


        
        self.samples = []
        for item in self.raw_data:
           
            profile_prompts = []
            
            p = item["profile"][0]

            desc = p.get("description",  "") or ""
            review = p.get("reviewText",  "") or ""
            output = item["output"]
            profile_prompts.append(f"Past item description:\n{desc}\n Past review: {review}\n\n")


            # LLaMA user prompt
            input_prompt = (
                "Previous review:\n\n"
                "<BERT>\n\n"
                "Based on the user's previous review and user model provided above, "
          "please complete the following instruction as if you were the user. Write at least 300 words."
                + item['input'] +
                "\nOnly output your review alone. Write at least 300 words!!!\n"
            )

            self.samples.append({
                "profile_prompts": profile_prompts,   # len(list[str])  == num_qa
                "input_prompt": input_prompt, 
                "output_text": output
            })

    def _load_data(self,  path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"data file does not esist: {path}")
        with open(path,  "r",  encoding="utf-8") as f:
            return json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,  idx):
        item = self.samples[idx]
        self.bert_tokenizer.padding_side = "left"
        self.llama_tokenizer.padding_side = "left"

        profile_prompts = item["profile_prompts"]
        input_prompt = item["input_prompt"]
        output_text = item["output_text"]

        # ===== 1) BERT 对 profile（当作 QA 条目）逐条编码, 并记录 "review:" 的 token 起始位置 =====
        # BERT encode profile and record the start position of review, which will be used to construct user model
        qa_input_ids_list = []
        qa_attention_masks_list = []
        title_start_indices_list = []

        # 预先准备候选子词序列, 增强健壮性
        # prepare target sequence
        target_variants = ["Past review:\n"]
        target_subtokens_variants = [self.bert_tokenizer.tokenize(tv) for tv in target_variants]

        for prof_text in profile_prompts:
            enc = self.bert_tokenizer(
                prof_text, 
                max_length=self.bert_max_len, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt", 
                add_special_tokens=True
            )
            ids = enc["input_ids"].squeeze(0)
            att = enc["attention_mask"].squeeze(0)

            # 在 token 序列中查找 "review" 的起始 token 位置
            # find review start idx in token sequence
            tokens = self.bert_tokenizer.convert_ids_to_tokens(ids)
            found_idx = -1
            for target_subtokens in target_subtokens_variants:
                if not target_subtokens:
                    continue
                tlen = len(target_subtokens)
                for i in range(len(tokens) - tlen + 1):
                    if tokens[i:i + tlen] == target_subtokens:
                        found_idx = i +tlen  
                        break
                if found_idx != -1:
                    break

            qa_input_ids_list.append(ids)
            qa_attention_masks_list.append(att)
            title_start_indices_list.append(found_idx)

        # 堆叠成 [num_qa,  L] stack
        bert_qa_id = torch.stack(qa_input_ids_list)
        bert_qa_atten = torch.stack(qa_attention_masks_list)
        answer_start_indices = torch.tensor(title_start_indices_list,  dtype=torch.long)

        # ===== 2) 构造 LLaMA 的 chat prompt（system + user + assistant）, 并编码 =====
        #build LLaMA chat template and encode
        chat = [
            {"role": "system",  "content": "You are a helpful assistant good at writing review based on description."}, 
            {"role": "user",  "content": input_prompt}, 
        ]
        full_prompt = self.llama_tokenizer.apply_chat_template(
            chat,  tokenize=False,  add_generation_prompt=True
        )

        question_enc = self.llama_tokenizer(
            full_prompt, 
            max_length=self.llama_max_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        question_ids = question_enc["input_ids"].squeeze(0)
        question_attention_mask = question_enc["attention_mask"].squeeze(0)

        # ===== 3) labels：只编码答案文本 =====
        # labels: encode answer text only
        label_enc = self.llama_tokenizer(
            output_text.strip() + self.llama_tokenizer.eos_token, 
            max_length=self.llama_max_answer_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        labels = label_enc["input_ids"].squeeze(0)

        return {
            # BERT 编码（与 UBC_data 对齐的命名）        bert encode
            "bert_qa_id": bert_qa_id,                      # [num_qa,  L_bert]
            "bert_qa_atten": bert_qa_atten,                # [num_qa,  L_bert]
            "answer_start_indices": answer_start_indices,  # [num_qa] —— "review"  token start idx（not found -1）
            # LLaMA 编码（与 UBC_data 对齐的命名）       llama encode
            "question_ids": question_ids, 
            "question_attention_mask": question_attention_mask, 
            "labels": labels
        }
