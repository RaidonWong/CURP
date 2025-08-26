import json
import os
import torch
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==================== Configuration ====================
raw_path = "/root/wangliang/Understanding-Generation/data/lamp7_val.json"
answer_path = "/root/wangliang/Understanding-Generation/data/lamp7_val_answer.json"

model_path = "/remote-home/share/lwang_share/models/LLM-Research/Meta-Llama-3-8B-Instruct_pad_bert/"

output_dir = "/root/wangliang/Understanding-Generation/curp/CSG/Tweet_paraphrase"
os.makedirs(output_dir, exist_ok=True)

filtered_data_file = f"{output_dir}/val_filtered.json"
judgment_file = f"{output_dir}/consistency_judgments.jsonl"
final_output_file = f"{output_dir}/val_final.json"

device = "cuda:0"
batch_size = 64
random.seed(42)  # For reproducibility

# ==================== Step 1: Load raw and answer data, filter short outputs ====================
print(" Step 1: Loading and filtering raw data (removing short outputs)...")

with open(raw_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
with open(answer_path, 'r', encoding='utf-8') as f:
    answer_data = json.load(f)

# Create answer map: id -> output
answer_map = {item["id"]: item["output"] for item in answer_data.get("golds", [])}

filtered_samples = []
for item in raw_data:
    sid = item["id"]
    if sid not in answer_map:
        continue
    output = answer_map[sid].strip()
    original_input = item.get("input", "").strip()

    # Filter out very short outputs or missing inputs
    if len(output.split()) < 4 or not original_input:
        continue

    new_item = {
        "id": sid,
        "input": original_input,
        "output": output,
        "profile": item.get("profile", [])[:4]  # Keep limited profile for completeness
    }
    filtered_samples.append(new_item)

# Save filtered data
with open(filtered_data_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_samples, f, ensure_ascii=False, indent=2)
print(f" Kept {len(filtered_samples)} samples with output ≥4 words → {filtered_data_file}")

# ==================== Step 2: Load model ====================
print(" Step 2: Loading Llama-3 model for paraphrase judgment...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = "<PAD>"
tokenizer.padding_side = "left"

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

# ==================== Prompt template: Judge if it's a valid paraphrase ====================
def make_prompt(original_tweet, paraphrased_tweet):
    return f"""
Is the second tweet a valid paraphrase of the first tweet?
This means they should convey the same core meaning. Answer only "Yes" or "No".

Original Tweet:
{original_tweet.strip()}

Paraphrased Tweet:
{paraphrased_tweet.strip()}

Your answer is:
Answer only "Yes" or "No"
""".strip()

# ==================== Batch inference ====================
print(" Step 3: Running batch inference for paraphrase consistency...")

with open(judgment_file, 'w', encoding='utf-8') as out_f:
    for i in tqdm(range(0, len(filtered_samples), batch_size), desc="Judging Paraphrase Consistency"):
        batch = filtered_samples[i:i + batch_size]
        prompts = []
        metadata = []

        for item in batch:
            prompt = make_prompt(item["input"], item["output"])
            prompts.append(prompt)
            metadata.append({
                "id": item["id"],
                "original_tweet": item["input"],
                "paraphrased_tweet": item["output"]
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
                    max_new_tokens=40,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, input_len:]

            for j in range(len(batch)):
                raw_response = tokenizer.decode(generated_ids[j], skip_special_tokens=True).strip()
                result = {
                    "id": metadata[j]["id"],
                    "original_tweet": metadata[j]["original_tweet"],
                    "paraphrased_tweet": metadata[j]["paraphrased_tweet"],
                    "raw_response": raw_response
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

        except Exception as e:
            print(f" Error in batch {i}: {e}")
            for meta in metadata:
                result = {
                    "id": meta["id"],
                    "original_tweet": meta["original_tweet"],
                    "paraphrased_tweet": meta["paraphrased_tweet"],
                    "raw_response": f"[ERROR] {str(e)}"
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

print(f" Large model judgment completed → {judgment_file}")

# ==================== Step 4: Extract 'Yes' responses, sample, and sort by original order ====================
print(" Step 4: Filtering 'Yes' responses and sampling...")

yes_indices = []
with open(judgment_file, 'r', encoding='utf-8') as f:
    for line_idx, line in enumerate(f):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            if "yes" in data.get("raw_response", "").lower():
                yes_indices.append(line_idx)
        except Exception as e:
            print(f"Failed to parse line {line_idx}: {e}")
            continue

print(f" Found {len(yes_indices)} samples with 'Yes' response.")

# Map yes_indices to actual filtered_samples indices
valid_indices = [i for i in yes_indices if i < len(filtered_samples)]
selected_with_index = [
    {"data": filtered_samples[i], "original_index": i} for i in valid_indices
]

sample_size = 1200
if len(selected_with_index) <= sample_size:
    sampled_with_index = selected_with_index
    print(f"  Less than {sample_size} valid samples. Keeping all {len(selected_with_index)} samples.")
else:
    sampled_with_index = random.sample(selected_with_index, sample_size)
    print(f" Randomly sampled {sample_size} items (seed=42).")

# Sort by original index to maintain input order
sampled_with_index.sort(key=lambda x: x["original_index"])
final_data = [item["data"] for item in sampled_with_index]

# ==================== Save final dataset ====================
with open(final_output_file, 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

print(f"   Full pipeline completed!")
print(f"   Final dataset saved to: {final_output_file}")
print(f"   Total {len(final_data)} samples, sorted by original order, ready for CSG_tweet task.")