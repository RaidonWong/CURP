import torch
from transformers import AutoTokenizer, RobertaTokenizer
from curp.model import SSA_model
from utils.utils import setup_seed
from curp.dataset import SSA_infer, simple_collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os


def load_model(config, device):
    """
    Load the SSA_model with pre-trained weights.
    
    Args:
        config (dict): Configuration dictionary.
        device (torch.device): Device to load the model on.
    
    Returns:
        model, bert_tokenizer, llama_tokenizer
    """
    # Initialize tokenizers
    bert_tokenizer = RobertaTokenizer.from_pretrained(config["bert_model_name_or_path"])
    llama_tokenizer = AutoTokenizer.from_pretrained(config["llama_model_name_or_path"])
    llama_tokenizer.pad_token = "<PAD>"
    llama_tokenizer.padding_side = "left"
    bert_tokenizer.padding_side = "left"

    # Build model
    model = SSA_model(
        encoder_path=config["bert_model_name_or_path"],     # BERT model path
        decoder_path=config["llama_model_name_or_path"],    # LLaMA model path
        freeze_bert=True,                                   # Freeze BERT encoder
        bert_final_seq_len=150                              # Max sequence length for BERT
    )
    model = model.half()  # Convert to FP16

    # Load trained weights
    model_weights_path = os.path.join(config["model_save_path"], config["model_weights_file"])
    state_dict = torch.load(model_weights_path, map_location=device)

    # Filter to include only mlp1, stage1, and stage2 parameters
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if 'mlp1' in k or 'stage1' in k or 'stage2' in k
    }

    # Load filtered weights
    model.load_state_dict(filtered_state_dict, strict=False)
    print("Model weights have been loaded successfully.")

    model.to(device)
    model.eval()
    return model, bert_tokenizer, llama_tokenizer


def predict(model, bert_tokenizer, llama_tokenizer, data_loader, device, max_new_tokens=150):
    """
    Run inference and generate answers.
    
    Returns:
        List of raw questions, generated answers, and reference labels.
    """
    generated_answers = []
    raw_questions = []
    reference_labels = []

    for batch in tqdm(data_loader, desc="Generating responses"):
        # Move inputs to device
        encoder_input = batch['bert_input_ids'].to(device)
        encoder_input_atten = batch['bert_attention_mask'].to(device)
        question_ids = batch['question_ids'].to(device)
        question_attention_mask = batch['question_attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model.generate_answer(
                encoder_input=encoder_input,
                encoder_input_atten=encoder_input_atten,
                question_ids=question_ids,
                question_attention_mask=question_attention_mask,
                max_new_tokens=max_new_tokens,
            )

        # Decode and store results
        for i in range(len(outputs)):
            decoded_label = llama_tokenizer.decode(labels[i], skip_special_tokens=True).strip()
            decoded_answer = llama_tokenizer.decode(outputs[i], skip_special_tokens=True).strip()
            decoded_question = llama_tokenizer.decode(question_ids[i], skip_special_tokens=True).strip()

            reference_labels.append(decoded_label)
            generated_answers.append(decoded_answer)
            raw_questions.append(decoded_question)

    return raw_questions, generated_answers, reference_labels


def inference_from_dataset(config, device, dataset_path):
    """
    Load dataset, run inference, and save results.
    """
    # Load model and tokenizers
    model, bert_tokenizer, llama_tokenizer = load_model(config, device)

    # Load dataset
    dataset = SSA_infer(
        data_path=dataset_path,
        encoder=bert_tokenizer,
        decoder=llama_tokenizer,
        decoder_max_question_len=180,   # Max input length for LLaMA
        decoder_max_answer_len=150      # Max output length for LLaMA
    )

    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=20,
        shuffle=False,
        collate_fn=simple_collate_fn,
        num_workers=4
    )

    # Run prediction
    questions, generated_answers, reference_labels = predict(
        model=model,
        bert_tokenizer=bert_tokenizer,
        llama_tokenizer=llama_tokenizer,
        data_loader=data_loader,
        device=device,
        max_new_tokens=150
    )

    # Save results
    output_path = "/root/wangliang/Understanding-Generation/curp/ssa_infer.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (label, pred) in enumerate(zip(reference_labels, generated_answers)):
            entry = {
                "id": i,
                "question": label,
                "answer": pred
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Inference completed. Results saved to {output_path}")


def main():
    # Configuration
    config = {
        "device": torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
        "llama_model_name_or_path": "/remote-home/share/lwang_share/models/LLM-Research/Meta-Llama-3-8B-Instruct_pad_bert/",
        "bert_model_name_or_path": "/remote-home/share/lwang_share/models/AI-ModelScope/roberta-large/",
        "model_save_path": "/remote-home/share/lwang_share/understand_generate/curp_ssa_llama3/",
        "model_weights_file": "lora_mlp_model_weights.pth",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.1,
        "lora_bias": "none",
        "seed": 42,
    }

    # Set random seed for reproducibility
    setup_seed(config["seed"])
    device = config["device"]

    # Dataset path
    dataset_path = "/remote-home/share/lwang_share/data/filtered_dataset_val_100_128_top500.json"

    # Start inference
    inference_from_dataset(config, device, dataset_path)


if __name__ == "__main__":
    main()