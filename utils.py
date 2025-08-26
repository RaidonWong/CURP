import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

from accelerate import Accelerator
from accelerate.utils import tqdm



import random
import numpy as np
import os


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed is set to {seed}")



# ---  train_one_epoch ---
def train_one_epoch_accelerate(accelerator, model, dataloader, optimizer):
    model.train()
    total_loss = 0
    #  accelerator.gradient_accumulation_steps 
    accumulation_steps = accelerator.gradient_accumulation_steps

    # 使用 accelerate 的 tqdm
    progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process, desc="Training")

    for step, batch in enumerate(dataloader):
        # accelerator.prepare(dataloader) automedically move the data to right device 


        with accelerator.accumulate(model): # 梯度累积上下文 gradient accumulate context
            # 前向传播 Forward prppogation 
            outputs = model(
                encoder_input=batch['bert_input_ids'],
                encoder_input_atten=batch['bert_attention_mask'],
                question_ids=batch['question_ids'],
                question_attention_mask=batch['question_attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss


            # 1. 计算平均损失 (用于梯度累积)
            # calculate average loss (for gradient accumulate)
            avg_loss = loss / accumulation_steps

            # 2. 反向传播 (使用 accelerator)
            # BP with accelerate
            accelerator.backward(avg_loss)


            if accelerator.is_main_process: # 可以只在主进程更新进度条 update progress_bar only in main process
                  progress_bar.update(1)
                  progress_bar.set_postfix({"loss": loss.item()}) # 显示原始 batch loss  show original batch loss

            # 优化器步骤和梯度清零由 accelerator.accumulate 处理
            # optimization and empty_cache will be dealed by accelerator.accumulate


            total_loss += loss.item() 

    # 计算整个 epoch 的平均 loss       calculate average loss in the whole epoch

    avg_epoch_loss = total_loss / len(dataloader)



    return avg_epoch_loss 






def train_one_epoch_accelerate_ubc(accelerator, model, dataloader, optimizer):
    model.train()
    total_loss = 0

    accumulation_steps = accelerator.gradient_accumulation_steps

    # 使用 accelerate 的 tqdm
    progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process, desc="traning")

    for step, batch in enumerate(dataloader):

        with accelerator.accumulate(model): 
            outputs = model(
                bert_profile_id=batch['bert_profile_id'],
                bert_profile_atten=batch['bert_profile_atten'],
                bert_qa_id=batch['bert_qa_id'],
                bert_qa_atten=batch['bert_qa_atten'],
                answer_start_indices=batch['answer_start_indices'],
                question_ids=batch['question_ids'],
                question_attention_mask=batch['question_attention_mask'],
                labels=batch['labels']
                )
            loss = outputs.loss


            avg_loss = loss / accumulation_steps


            accelerator.backward(avg_loss)


            if accelerator.is_main_process: 
                  progress_bar.update(1)
                  progress_bar.set_postfix({"loss": loss.item()}) 



            total_loss += loss.item() 


    avg_epoch_loss = total_loss / len(dataloader)

    return avg_epoch_loss 



















def evaluate_accelerate(accelerator, model, dataloader):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process, desc="evaluating")

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                encoder_input=batch['bert_input_ids'],
                encoder_input_atten=batch['bert_attention_mask'],
                question_ids=batch['question_ids'],
                question_attention_mask=batch['question_attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss
            batch_size = batch['bert_input_ids'].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_postfix({"eval_loss": loss.item()})


    total_loss_tensor = torch.tensor(total_loss, device=accelerator.device)
    total_samples_tensor = torch.tensor(total_samples, device=accelerator.device)

    
    total_loss_tensor = accelerator.gather(total_loss_tensor)
    total_samples_tensor = accelerator.gather(total_samples_tensor)

    
    avg_loss = (total_loss_tensor.sum() / total_samples_tensor.sum()).item()
    accelerator.wait_for_everyone()
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss
    



def evaluate_accelerate_ubc(accelerator, model, dataloader):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process, desc="evaluating")

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                bert_profile_id=batch['bert_profile_id'],
                bert_profile_atten=batch['bert_profile_atten'],
                bert_qa_id=batch['bert_qa_id'],
                bert_qa_atten=batch['bert_qa_atten'],
                answer_start_indices=batch['answer_start_indices'],
                question_ids=batch['question_ids'],
                question_attention_mask=batch['question_attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss
            batch_size = batch['bert_profile_id'].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_postfix({"eval_loss": loss.item()})


    total_loss_tensor = torch.tensor(total_loss, device=accelerator.device)
    total_samples_tensor = torch.tensor(total_samples, device=accelerator.device)

    total_loss_tensor = accelerator.gather(total_loss_tensor)
    total_samples_tensor = accelerator.gather(total_samples_tensor)

    
    avg_loss = (total_loss_tensor.sum() / total_samples_tensor.sum()).item()
    accelerator.wait_for_everyone()
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss
    

