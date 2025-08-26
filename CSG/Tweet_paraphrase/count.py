import json
import os
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# 配置路径
bert_model_path = "/remote-home/share/lwang_share/models/AI-ModelScope/roberta-large/"
qwen_model_path = "/remote-home/share/lwang_share/models/AI-ModelScope/Qwen3-Embedding-0.6B/"
input_file = "/root/wangliang/Understanding-Generation/curp/CSG/Tweet_paraphrase/filtered_ans.json"

# 设置设备
device = "cuda:1" if torch.cuda.is_available() else "cpu"

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件未找到: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in tqdm(zip(predictions, references), desc="Calculating ROUGE", total=len(predictions)):
        score = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(score[key].fmeasure)
    avg_scores = {k: sum(v) / len(v) for k, v in scores.items()}
    return avg_scores

def compute_rouge_individual(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    rouge1_scores = []
    for pred, ref in tqdm(zip(predictions, references), desc="Calculating ROUGE-1 for filtering", total=len(predictions)):
        score = scorer.score(ref, pred)
        rouge1_scores.append(score['rouge1'].fmeasure)
    return rouge1_scores

def compute_bleu(predictions, references):
    references_list = [[ref.split()] for ref in references]
    predictions_list = [pred.split() for pred in predictions]
    chencherry = SmoothingFunction()
    return corpus_bleu(
        references_list,
        predictions_list,
        smoothing_function=chencherry.method1
    )

def compute_cos_sim(predictions, references, model_path, strategy="cls", batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).to(device)
    model.eval()

    def get_embedding_batch(texts_batch):
        inputs = tokenizer(texts_batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            if strategy == "cls":
                return hidden[:, 0]
            elif strategy == "eos":
                eos_indices = inputs.attention_mask.sum(dim=1) - 1
                return hidden[torch.arange(len(texts_batch)), eos_indices]
            elif strategy == "mean":
                mask = inputs.attention_mask.unsqueeze(-1)
                sum_hidden = (hidden * mask).sum(dim=1)
                lengths = mask.sum(dim=1)
                return sum_hidden / lengths
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")

    all_cos_scores = []
    num_samples = len(predictions)

    for i in tqdm(range(0, num_samples, batch_size), desc=f"Calculating Cos-Sim ({strategy})"):
        pred_batch = predictions[i:i + batch_size]
        ref_batch = references[i:i + batch_size]

        pred_emb = get_embedding_batch(pred_batch)
        ref_emb = get_embedding_batch(ref_batch)

        cos_scores_batch = F.cosine_similarity(pred_emb, ref_emb, dim=-1)
        all_cos_scores.extend(cos_scores_batch.cpu().tolist())

    return sum(all_cos_scores) / len(all_cos_scores) if all_cos_scores else 0.0

def main():
    data = load_data(input_file)

    all_references = [item["gold"] for item in data]
    all_predictions = [item["prediction"] for item in data]

    print(f"📊 原始数据共 {len(all_references)} 条")

    # Step 1: 计算 ROUGE-1
    rouge1_scores = compute_rouge_individual(all_predictions, all_references)

    # Step 2: 过滤：长度 >= 4 且 ROUGE-1 > 0
    filtered_refs = []
    filtered_preds = []
    for pred, ref, rouge1 in zip(all_predictions, all_references, rouge1_scores):
        if len(pred.split()) >= 0 and len(ref.split()) >=0  and rouge1 >= 0.0:
            filtered_preds.append(pred)
            filtered_refs.append(ref)

    print(f"✅ 过滤后剩余 {len(filtered_refs)} 条数据")

    assert len(filtered_refs) == len(filtered_preds), "过滤后数据数量不一致！"

    print("🔄 正在计算 ROUGE...")
    rouge_scores = compute_rouge(filtered_preds, filtered_refs)

    print("🔄 正在计算 BLEU...")
    bleu_score = compute_bleu(filtered_preds, filtered_refs)

    embedding_batch_size = 32

    print(f"🔄 正在计算 RoBERTa 相似度 (CLS) (batch_size={embedding_batch_size})...")
    cos_sim_bert = compute_cos_sim(filtered_preds, filtered_refs, bert_model_path, strategy="cls", batch_size=embedding_batch_size)

    print(f"🔄 正在计算 Qwen3 相似度 (EOS) (batch_size={embedding_batch_size})...")
    cos_sim_qwen = compute_cos_sim(filtered_preds, filtered_refs, qwen_model_path, strategy="eos", batch_size=embedding_batch_size)

    print("\n📈 评估结果:")
    print(f" - ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f" - ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f" - ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print(f" - BLEU: {bleu_score:.4f}")
    print(f" - RoBERTa Cos-Sim (CLS): {cos_sim_bert:.4f}")
    print(f" - Qwen3 Cos-Sim (EOS): {cos_sim_qwen:.4f}")

if __name__ == "__main__":
    main()
