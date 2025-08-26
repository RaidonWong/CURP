import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import pickle
import os
from accelerate import Accelerator
from transformers import AutoTokenizer
from accelerate.utils import tqdm
from peft import get_peft_model_state_dict
from peft import get_peft_model, LoraConfig, TaskType
from curp.dataset import UBC_data,simple_collate_fn,UBC_infer
from curp.model import UBC_model
from curp.utils import setup_seed,train_one_epoch_accelerate_ubc,evaluate_accelerate_ubc

from tqdm import tqdm
import json


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"random seed set to {seed}")



def load_model(config, device):

    bert_tokenizer = AutoTokenizer.from_pretrained(config["bert_model_name_or_path"])
    llama_tokenizer = AutoTokenizer.from_pretrained(config["llama_model_name_or_path"])
    llama_tokenizer.pad_token = '<PAD>'
    llama_tokenizer.pad_token_id=128256
    llama_tokenizer.padding_side='left'
    bert_tokenizer.padding_side="left"

    # 构建模型
    model = UBC_model(
    encoder_path=config["bert_model_name_or_path"],  # BERT 
    decoder_path=config["llama_model_name_or_path"],  # llama 
    codebook_path=config["codebook_path"],
    lora_r=16,  
    lora_alpha=32,  
    lora_target_modules=config["lora_target_modules"],  
    lora_dropout=0.1,  
    lora_bias="none",  
    freeze_bert=True,  
    bert_final_seq_len=256 
)


    model_weights_path = os.path.join(config["model_save_path"], config["model_weights_file"])


    state_dict = torch.load(model_weights_path, map_location=device)


    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if 'mlp1' in k or 'stage1' in k or 'stage2' in k or 'lora' in k or 'codebook' in k
    }
    model.load_state_dict(filtered_state_dict, strict=False)



    model = model.half().to(device)
    print(f"model device: {next(model.parameters()).device}")
    model.eval()

    return model, bert_tokenizer, llama_tokenizer

from tqdm import tqdm
import torch

def predict(model, llama_tokenizer, data_loader, device, max_new_tokens=128):
    results = []
    pbar = tqdm(data_loader, desc="inferencing", total=len(data_loader))

    with torch.no_grad():
        for batch in pbar:

            bert_profile_id=batch['bert_profile_id'].to(device)
            bert_profile_atten=batch['bert_profile_atten'].to(device)
            bert_qa_id=batch['bert_qa_id'].to(device)
            bert_qa_atten=batch['bert_qa_atten'].to(device)
            answer_start_indices=batch['answer_start_indices'].to(device)
            question_ids=batch['question_ids'].to(device)
            question_attention_mask=batch['question_attention_mask'].to(device)
            labels=batch['labels']


            generated_ids = model.generate_answer(
                bert_profile_id=bert_profile_id,
                bert_profile_atten=bert_profile_atten,
                bert_qa_id=bert_qa_id,
                bert_qa_atten=bert_qa_atten,
                answer_start_indices=answer_start_indices,
                question_ids=question_ids,
                question_attention_mask=question_attention_mask,
                max_new_tokens=max_new_tokens,
            )


            decoded_questions = llama_tokenizer.batch_decode(question_ids, skip_special_tokens=True)
            decoded_preds = llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            filtered_labels = [
                label[label != -100] for label in labels
            ]


            decoded_ans = llama_tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)

            for i in range(len(decoded_questions)):
                results.append({
                    "question": decoded_questions[i],
                    "prediction": decoded_preds[i],
                    "answer":decoded_ans[i]
        
                })

    return results


def inference_from_dataset(config, device):

    model, bert_tokenizer, llama_tokenizer = load_model(config, device)


    val_dataset = UBC_infer(
    data_path=config["val_data_path"],  
    bert_tokenizer=bert_tokenizer,
    llama_tokenizer=llama_tokenizer,
    bert_max_len= 256,
    llama_max_question_len =360,
    llama_max_answer_len=128,
    qa_num=2
)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=simple_collate_fn,
        num_workers=4
    )


    print("Start inference...")
    results = predict(model, llama_tokenizer, val_dataloader, device, max_new_tokens=120)

    output_path = os.path.join(config["output_dir"], "curp_ubc.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"inference finished, save to {output_path}")


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    config = {
        "device": torch.device("cuda:2" if torch.cuda.is_available() else "cpu"),
        "llama_model_name_or_path": "/remote-home/share/lwang_share/models/LLM-Research/Meta-Llama-3-8B-Instruct_pad_bert/", # 替换为实际路径
        "bert_model_name_or_path": "/remote-home/share/lwang_share/models/AI-ModelScope/roberta-large/",  # 替换为实际路径
        "val_data_path": "/remote-home/share/lwang_share/data/filtered_data_128_256_test_longest_filtered.json",
        "codebook_path": "/remote-home/share/lwang_share/understand_generate/codebook/stage2_codebook_270.pt",
        "model_save_path": "/remote-home/share/lwang_share/understand_generate/curp_ubc/",
        "model_weights_file": "lora_mlp_model_weights.pth",
        "output_dir": "/root/wangliang/Understanding-Generation/curp",
        "batch_size": 64,
        "seed":42,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.1,
        "lora_bias": "none"
    }

    setup_seed(config["seed"])

    inference_from_dataset(config, device=config["device"])


if __name__ == "__main__":
    main()