import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from curp.model import CSG
from curp.dataset import CSG_news, simple_collate_fn
from tqdm import tqdm
import json


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed has been set to {seed}")


def load_model(config, device):
    # Initialize tokenizers
    bert_tokenizer = AutoTokenizer.from_pretrained(config["bert_model_name_or_path"])
    llama_tokenizer = AutoTokenizer.from_pretrained(config["llama_model_name_or_path"])
    llama_tokenizer.pad_token = '<PAD>'
    llama_tokenizer.pad_token_id = 128256
    llama_tokenizer.padding_side = 'left'

    # Build model
    model = CSG(
        encoder_path=config["bert_model_name_or_path"],  # BERT model path
        decoder_path=config["llama_model_name_or_path"],  # LLaMA model path
        codebook_path=config["codebook_path"],
        lora_r=16,  # Rank of LoRA adapters
        lora_alpha=32,  # Alpha value for LoRA
        lora_target_modules=config["lora_target_modules"],  # Modules to apply LoRA to
        lora_dropout=0.1,  # Dropout probability for LoRA
        lora_bias="none",  # Do not use LoRA bias
        freeze_bert=True,  # Freeze BERT encoder
        bert_final_seq_len=80  # Maximum sequence length for BERT
    )
    model = model.half()
    model = model.to(device)

    # Load model weights
    model_weights_path = os.path.join(config["model_save_path"], config["model_weights_file"])
    state_dict = torch.load(model_weights_path, map_location=device)

    # Filter state_dict to include only mlp1, stage1, stage2, lora, and codebook parameters
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if 'mlp1' in k or 'stage1' in k or 'stage2' in k or 'lora' in k or 'codebook' in k
    }

    # Load filtered weights into the model
    model.load_state_dict(filtered_state_dict, strict=False)

    # Get device of LLaMA input embeddings
    llama_input_embedding_device = model.llama.get_input_embeddings().weight.device
    print(f"LLaMA input embedding layer is on: {llama_input_embedding_device}")

    # Move custom modules (mlp1, stage1, stage2, codebook) to the same device as LLaMA embeddings and convert to FP16
    model.mlp1 = model.mlp1.half().to(llama_input_embedding_device)
    model.stage1 = model.stage1.half().to(llama_input_embedding_device)
    model.stage2 = model.stage2.half().to(llama_input_embedding_device)
    model.codebook = model.codebook.half().to(llama_input_embedding_device)

    model.eval()
    return model, bert_tokenizer, llama_tokenizer


def predict(model, llama_tokenizer, data_loader, device, max_new_tokens=20):
    results = []
    pbar = tqdm(data_loader, desc="Generating predictions", total=len(data_loader))
    llama_input_embedding_device = model.llama.get_input_embeddings().weight.device
    print(f"Batch data will be moved to: {llama_input_embedding_device}")

    with torch.no_grad():
        for batch in pbar:
            # Move all required inputs to the target device
            bert_qa_id = batch['bert_qa_id'].to(llama_input_embedding_device)
            bert_qa_atten = batch['bert_qa_atten'].to(llama_input_embedding_device)
            answer_start_indices = batch['answer_start_indices'].to(llama_input_embedding_device)
            question_ids = batch['question_ids'].to(llama_input_embedding_device)
            question_attention_mask = batch['question_attention_mask'].to(llama_input_embedding_device)
            labels = batch['labels'].to(llama_input_embedding_device)

            # Generate answers using the custom generate_answer method
            generated_ids = model.generate_answer(
                bert_qa_id=bert_qa_id,
                bert_qa_atten=bert_qa_atten,
                answer_start_indices=answer_start_indices,
                question_ids=question_ids,
                question_attention_mask=question_attention_mask,
                max_new_tokens=max_new_tokens,
            )

            # Decode predictions
            decoded_preds = llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Store results
            for i in range(len(decoded_preds)):
                results.append({
                    "prediction": decoded_preds[i],
                })

    return results


def inference_from_dataset(config, device):
    # Load model and tokenizers
    model, bert_tokenizer, llama_tokenizer = load_model(config, device)

    # Load validation dataset
    val_dataset = CSG_news(
        raw_path=config["val_data_path"],
        llama_tokenizer=llama_tokenizer,
        bert_tokenizer=bert_tokenizer,
        bert_max_len=80,
        llama_max_len=200,
        llama_max_answer_len=40,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=simple_collate_fn,
        num_workers=4
    )

    # Start inference
    print(" Starting inference...")
    results = predict(model, llama_tokenizer, val_dataloader, device, max_new_tokens=30)

    # Save results to file
    output_path = os.path.join(config["output_dir"], "csg_news.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f" Inference completed. Results saved to {output_path}")


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    config = {
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "llama_model_name_or_path": "/remote-home/share/lwang_share/models/LLM-Research/Meta-Llama-3-8B-Instruct_pad_bert/",
        "bert_model_name_or_path": "/remote-home/share/lwang_share/models/AI-ModelScope/roberta-large/",
        "val_data_path": "/root/wangliang/Understanding-Generation/curp/CSG/News_headline/val_final.json",
        "codebook_path": "/remote-home/share/lwang_share/understand_generate/codebook/stage2_codebook_270.pt",
        "model_save_path": "/remote-home/share/lwang_share/understand_generate/curp_ubc/",
        "model_weights_file": "lora_mlp_model_weights.pth",
        "output_dir": "/root/wangliang/Understanding-Generation/curp/CSG",
        "batch_size": 16,
        "seed": 42,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.1,
        "lora_bias": "none"
    }

    setup_seed(config["seed"])

    # Start inference
    inference_from_dataset(config, device=config["device"])


if __name__ == "__main__":
    main()