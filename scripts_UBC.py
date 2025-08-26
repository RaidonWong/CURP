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
from curp.dataset import UBC_data,simple_collate_fn
from curp.model import UBC_model
from curp.utils import setup_seed,train_one_epoch_accelerate_ubc,evaluate_accelerate_ubc


def main(config):
    # --- 初始化 Accelerator ---
    # gradient_accumulation_steps 可以从 config 或 accelerate config 读取
    accelerator = Accelerator(gradient_accumulation_steps=config["gradient_accumulation_steps"])


    setup_seed(config["seed"])

    # 设备由 Accelerator 管理
    device = accelerator.device
    print(f"进程 {accelerator.process_index} 使用设备: {device}")
    # --- load Tokenizers ---

    bert_tokenizer = AutoTokenizer.from_pretrained(config["bert_model_name_or_path"])
    llama_tokenizer = AutoTokenizer.from_pretrained(config["llama_model_name_or_path"])
    llama_tokenizer.pad_token = '<PAD>' 
    llama_tokenizer.padding_side="left"
    bert_tokenizer.padding_side='left'

    

    # ---  Dataset  ---
    train_dataset = UBC_data(
    data_path=config["train_data_path"], 
    bert_tokenizer=bert_tokenizer,
    llama_tokenizer=llama_tokenizer,
    bert_max_len= 256,
    llama_max_question_len =360,
    llama_max_answer_len=128,
    qa_num=2
)

    val_dataset = UBC_data(
    data_path=config["val_data_path"],  
    bert_tokenizer=bert_tokenizer,
    llama_tokenizer=llama_tokenizer,
    bert_max_len= 256,
    llama_max_question_len =360,
    llama_max_answer_len=128,
    qa_num=2
)
    # ---  DataLoader (use simple_collate_fn) ---
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=simple_collate_fn, num_workers=config.get("num_workers", 4))
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=simple_collate_fn, num_workers=config.get("num_workers", 4))

    # --- initialize model ---
    model = UBC_model(
    encoder_path=config["bert_model_name_or_path"],  # BERT 
    decoder_path=config["llama_model_name_or_path"],  # llama 
    codebook_path=config["codebook_path"],
    lora_r=16,  # LoRA  rank
    lora_alpha=32,  # LoRA  alpha
    lora_target_modules=config["lora_target_modules"],  # lora module
    lora_dropout=0.1,  # LoRA dropout 
    lora_bias="none",  
    freeze_bert=True,  # freeze BERT 
    bert_final_seq_len=256  # BERT max sequense length
)
    model_weights_path = os.path.join(config["model_save_path"], config["model_weights_file"])

    # load full state_dict
    state_dict = torch.load(model_weights_path, map_location=device)
    # filter effective parts
    filtered_state_dict = {k: v for k, v in state_dict.items() if 'mlp1' in k or 'codebook' in k or 'stage1' in k or 'stage2' in k}

    # load state_dict
    model.load_state_dict(filtered_state_dict, strict=False)
    model = model.half()  # float 16
    model.to(accelerator.device)  

    codebook_params = []
    lora_params = []
    other_params = []

# group by name, each group with different learning rate
    for name, param in model.named_parameters():
        if "codebook.codebook" in name:  
            codebook_params.append(param)
        elif "lora" in name:  
            lora_params.append(param)
        else:
            other_params.append(param)

    optimizer_grouped_parameters = [
    {
        "params": codebook_params,
        "lr": 1e-3,  
        "weight_decay": 0.01,  
    },
    {
        "params": lora_params,
        "lr": 2e-5,  
        "weight_decay": 0.01, 
    },
    {
        "params": other_params,
        "lr": 5e-4,  
        "weight_decay": 0.01,
    }
]

    optimizer = optim.AdamW(optimizer_grouped_parameters)

    # ---  accelerator.prepare()  ---
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    # --- training ---
    best_val_loss = float('inf')
    for epoch in range(config["num_epochs"]):
        # print epoch information only in main process
        if accelerator.is_main_process:   
             print(f"\n---  {epoch+1} epoch /{config['num_epochs']}  ---")

        
        train_loss = train_one_epoch_accelerate_ubc(accelerator, model, train_dataloader, optimizer)
        if accelerator.is_main_process:
            print(f" epoch {epoch+1} , Train Loss: {train_loss:.4f}") 

        # Eval

        # --- Inside the training loop, after validation epoch ---
        if config["val_data_path"] and (epoch + 1) % config["eval_interval"] == 0:
            if accelerator.is_main_process:
                print(f"\n--- 在验证集上评估 (第 {epoch+1} 轮) ---")

            # 1. eval - make sure evaluate_accelerate return the same loss across all process
            val_loss = evaluate_accelerate_ubc(accelerator, model, val_dataloader) 


            # 2. make sure every process finish the eval
            accelerator.wait_for_everyone() 

            # 3. decide whether to save by main process，and update best_val_loss
            should_save_on_main = False
            if accelerator.is_main_process:
                print(f"process {accelerator.process_index} eval finish. Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}")
                if val_loss < best_val_loss:
                    should_save_on_main = True
                    print(f"val loss improve ({best_val_loss:.4f} --> {val_loss:.4f}). ready to save...")
                    best_val_loss = val_loss 
                else:
                    print(f"val loss {val_loss:.4f} do not improve from {best_val_loss:.4f}  jump the save")
           # 4. 将保存决策从主进程广播到所有进程  broadcast save decision from main to all process
            #    创建一个包含决策的列表 (或张量)，注意设备      build a dicision list
            #    使用 broadcast_object_list 是处理 Python 对象（如 bool）的简便方法      use broadcast_object_list
            decision_list = [should_save_on_main] 
            # torch.distributed.broadcast_object_list 需要知道哪个是源进程 (通常是 rank 0)        torch.distributed.broadcast_object_list need to know who is the main

            torch.distributed.broadcast_object_list(decision_list, src=0) 
            
 
            should_save_globally = decision_list[0]
            
            if should_save_globally:
                 # 所有进程必须一起调用 save_state          all process save stage at the same time
                save_dir = os.path.join(config["model_save_path"], f"{config['stage']}")
                print(f"Process {accelerator.process_index} (save) calling accelerator.save_state() to {save_dir}")
                try:

                    if accelerator.is_main_process:
                        

                        unwrapped_model = accelerator.unwrap_model(model)
                        

                        print(unwrapped_model)
                        save_weights_path = os.path.join(save_dir, 'pytorch_model.bin')
                        # get state_dict
                        state_dict = unwrapped_model.state_dict()

                         # only preserve effective parameters
                        filtered_state_dict = {}

                        for name, param in state_dict.items():
                            
                            if 'lora' in name or 'mlp1' in name or 'mlp2' in name or 'codebook' in name or 'stage1' in name or 'stage2' in name:  
                                filtered_state_dict[name] = param
                      
                        save_weights_path = os.path.join(save_dir, 'lora_mlp_model_weights.pth')
                        torch.save(filtered_state_dict, save_weights_path)
                        print(f"main process {accelerator.process_index} has saved parameters to {save_weights_path}")

                        tokenizer_save_path = os.path.join(save_dir, "tokenizer")
                        llama_tokenizer.save_pretrained(tokenizer_save_path)
                        print(f"have saved tokenizer to {tokenizer_save_path}")

                except Exception as e:
                    print(f"process {accelerator.process_index} failed during save_state : {e}")
                    # Consider raising the exception to stop training if saving fails critically
                
   
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    print("all process finish saving")

            else:

                if not accelerator.is_main_process:
                    print(f"Process {accelerator.process_index}(not save)  jump preserve")

            print(f"process {accelerator.process_index} finish eval or save")


    if accelerator.is_main_process:
        print("\nTraining finished.")
        if config["val_data_path"]:
            print(f"Best validation loss achieved: {best_val_loss:.4f}")




if __name__ == "__main__":

    config = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "gradient_accumulation_steps":1,
        "llama_model_name_or_path": "/remote-home/share/lwang_share/models/LLM-Research/Meta-Llama-3-8B-Instruct_pad_bert/", 
        "bert_model_name_or_path": "/remote-home/share/lwang_share/models/AI-ModelScope/roberta-large/",  
       
        "train_data_path": "/remote-home/share/lwang_share/data/filtered_data_128_256_train_longest_filtered.json",
        "val_data_path": "/remote-home/share/lwang_share/data/filtered_data_128_256_val_longest_filtered.json",
        "model_save_path": "/remote-home/share/lwang_share/understand_generate/",  
        "model_weights_file":"curp_ssa_llama3_2/lora_mlp_model_weights.pth",
        "codebook_path":"/remote-home/share/lwang_share/understand_generate/codebook/stage2_codebook_270.pt",
        "max_answer_len": 128,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "num_epochs": 3,
        "eval_interval": 1,  
        "seed": 42,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj"],
        "lora_bias": "none",
        "stage":"curp_ubc"
        

    }

    main(config)