import json
import os
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModel

# ==================== Configuration ====================
bert_model_path = "/remote-home/share/lwang_share/models/AI-ModelScope/roberta-large/"
qwen_model_path = "/remote-home/share/lwang_share/models/AI-ModelScope/Qwen3-Embedding-0.6B/"
input_file = "/root/wangliang/Understanding-Generation/curp/ssa_infer.json"

device = "cuda:1" if torch.cuda.is_available() else "cpu"


def load_data(path):
    """Load JSON data from file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_rouge(predictions, references):
    """Compute average ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(score[key].fmeasure)
    avg_scores = {k: sum(v) / len(v) for k, v in scores.items()}
    return avg_scores


def compute_bleu(predictions, references):
    """Compute corpus-level BLEU score with smoothing."""
    references_list = [[ref.split()] for ref in references]
    predictions_list = [pred.split() for pred in predictions]
    chencherry = SmoothingFunction()
    return corpus_bleu(
        references_list,
        predictions_list,
        smoothing_function=chencherry.method1
    )


def compute_cos_sim(predictions, references, model_path, strategy="cls"):
    """
    Compute mean cosine similarity between prediction and reference embeddings.
    
    Args:
        predictions: List of predicted texts.
        references: List of reference texts.
        model_path: Path to the pre-trained embedding model.
        strategy: How to pool embeddings ("cls", "eos", or "mean").
    
    Returns:
        Mean cosine similarity (float).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()

    def get_embedding(texts, batch_size=32):
        """Encode a list of texts into embeddings."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=150
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                hidden = outputs.last_hidden_state  # (B, L, H)

                if strategy == "cls":
                    emb = hidden[:, 0]  # First token (CLS)
                elif strategy == "eos":
                    # Last non-padding token
                    eos_indices = inputs.attention_mask.sum(dim=1) - 1
                    emb = hidden[torch.arange(len(batch)), eos_indices]
                elif strategy == "mean":
                    # Mean pooling over non-padding tokens
                    mask = inputs.attention_mask.unsqueeze(-1)
                    sum_hidden = (hidden * mask).sum(dim=1)
                    lengths = mask.sum(dim=1)
                    emb = sum_hidden / lengths
                else:
                    raise ValueError(f"Unsupported pooling strategy: {strategy}")
            embeddings.append(emb.cpu())
        return torch.cat(embeddings, dim=0).to(device)

    pred_emb = get_embedding(predictions)
    ref_emb = get_embedding(references)
    cos_scores = F.cosine_similarity(pred_emb, ref_emb, dim=-1)
    return cos_scores.mean().item()


def main():
    print(" Loading evaluation data...")
    data = load_data(input_file)

    references = [item["question"] for item in data]
    predictions = [item["answer"] for item in data]

    assert len(references) == len(predictions), "Mismatch in number of references and predictions!"

    print(f" Loaded {len(references)} samples for evaluation.\n")

    print(" Computing ROUGE scores...")
    rouge_scores = compute_rouge(predictions, references)

    print(" Computing BLEU score...")
    bleu_score = compute_bleu(predictions, references)

    print(" Computing RoBERTa embedding similarity (CLS pooling)...")
    cos_sim_bert = compute_cos_sim(predictions, references, bert_model_path, strategy="cls")

    print(" Computing Qwen3 embedding similarity (EOS pooling)...")
    cos_sim_qwen = compute_cos_sim(predictions, references, qwen_model_path, strategy="eos")

    # Final Results
    print("\n Final Evaluation Results:")
    print(f" - ROUGE-1:     {rouge_scores['rouge1']:.4f}")
    print(f" - ROUGE-2:     {rouge_scores['rouge2']:.4f}")
    print(f" - ROUGE-L:     {rouge_scores['rougeL']:.4f}")
    print(f" - BLEU:        {bleu_score:.4f}")
    print(f" - Cos-Sim (RoBERTa, CLS): {cos_sim_bert:.4f}")
    print(f" - Cos-Sim (Qwen3, EOS):   {cos_sim_qwen:.4f}")


if __name__ == "__main__":
    main()