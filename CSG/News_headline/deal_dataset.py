import json
import os
import torch
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==================== 配置路径 ====================
raw_path = "/root/wangliang/Understanding-Generation/data/lamp4_val.json"
answer_path = "/root/wangliang/Understanding-Generation/data/lamp4_val_answer.json"
model_path = "/remote-home/share/lwang_share/models/LLM-Research/Meta-Llama-3-8B-Instruct_pad_bert/"

output_dir = "/root/wangliang/Understanding-Generation/curp/CSG/News_headline"
os.makedirs(output_dir, exist_ok=True)

filtered_data_file = f"{output_dir}/val_filtered.json"
judgment_file = f"{output_dir}/consistency_judgments.jsonl"
final_output_file = f"{output_dir}/val_final.json"

device = "cuda:0"
batch_size = 64
random.seed(42)  


print(" Step 1: Loading and filtering raw data (remove short outputs)...")
with open(raw_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
with open(answer_path, 'r', encoding='utf-8') as f:
    answer_data = json.load(f)


answer_map = {item["id"]: item["output"] for item in answer_data.get("golds", [])}

filtered_samples = []
for item in raw_data:
    sid = item["id"]
    if sid not in answer_map:
        continue
    output = answer_map[sid].strip()
    if len(output.split()) < 4:  
        continue
    new_item = {k: v for k, v in item.items() if k != "profile"}
    new_item["output"] = output
    new_item["profile"] = item.get("profile", [])[:4]  
    filtered_samples.append(new_item)


with open(filtered_data_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_samples, f, ensure_ascii=False, indent=2)
print(f" Keep {len(filtered_samples)}  output length ≥4 samples → {filtered_data_file}")


print(" Step 2: Loading Llama-3 model for consistency judgment...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token="<PAD>"
tokenizer.padding_side="left"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=None
)
model.to(device)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

def make_prompt(article_excerpt, headline):
    return f"""
You are given the opening paragraph of a news article and a proposed headline.
Determine whether the headline is thematically related to the content of the paragraph — that is, whether it is plausible that such a headline could be generated based on this paragraph.
It is acceptable if the headline generalizes, predicts, or summarizes the main topic, as long as it is clearly connected to the key subject of the paragraph.
Do not expect full detail match; focus on topical relevance.

Opening Paragraph:
{article_excerpt.strip()}

Proposed Headline:
{headline.strip()}

Is the headline plausibly related to the opening paragraph? Please explain briefly and conclude with 'Yes' or 'No'.
""".strip()


print(" Step 3: Running batch inference...")

with open(judgment_file, 'w', encoding='utf-8') as out_f:
    for i in tqdm(range(0, len(filtered_samples), batch_size), desc="Judging Consistency"):
        batch = filtered_samples[i:i + batch_size]
        prompts = []
        metadata = []

        for item in batch:
            input_text = item["input"]
            prefix = "Generate a headline for the following article: "
            article_excerpt = input_text[len(prefix):].strip() if input_text.startswith(prefix) else input_text.strip()
            headline = item["output"]
            prompt = make_prompt(article_excerpt, headline)
            prompts.append(prompt)
            metadata.append({
                "id": item["id"],
                "article_excerpt": article_excerpt,
                "headline": headline
            })

        try:
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=400,
                add_special_tokens=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, input_len:]

            for j in range(len(batch)):
                raw_response = tokenizer.decode(generated_ids[j], skip_special_tokens=True).strip()
                result = {
                    "id": metadata[j]["id"],
                    "article_excerpt": metadata[j]["article_excerpt"],
                    "headline": metadata[j]["headline"],
                    "raw_response": raw_response
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

        except Exception as e:
            print(f"\n Error in batch {i}: {e}")
            for meta in metadata:
                result = {
                    "id": meta["id"],
                    "article_excerpt": meta["article_excerpt"],
                    "headline": meta["headline"],
                    "raw_response": f"[ERROR] {str(e)}"
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

print(f"LLM judge finish → {judgment_file}")


print("Step 4: Filtering 'Yes' responses and sampling...")


yes_indices = []
with open(judgment_file, 'r', encoding='utf-8') as f:
    for line_idx, line in enumerate(f):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            if "yes" in data.get("raw_response", "").lower():
                yes_indices.append(line_idx)
        except:
            continue

print(f" Find {len(yes_indices)} samples including yes ")


valid_indices = [i for i in yes_indices if i < len(filtered_samples)]
selected_with_index = [
    {"data": filtered_samples[i], "original_index": i} for i in valid_indices
]


sample_size = 1200
if len(selected_with_index) <= sample_size:
    sampled_with_index = selected_with_index
    print(f" data insufficient {sample_size} , keep all {len(selected_with_index)} samples")
else:
    sampled_with_index = random.sample(selected_with_index, sample_size)
    print(f"random sample {sample_size} ")


sampled_with_index.sort(key=lambda x: x["original_index"])
final_data = [item["data"] for item in sampled_with_index]


with open(final_output_file, 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

print(f"All finish, data saved to")
print(f"   {final_output_file}")
print(f"   in total {len(final_data)} ordered by original")