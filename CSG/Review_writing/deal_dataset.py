import json
import random
from transformers import AutoTokenizer

# Configuration paths
input_file = "/root/wangliang/Understanding-Generation/longlamp/val.json"
output_file = "/root/wangliang/Understanding-Generation/curp/CSG/Review_writing/val_filtered.json"
model_path = "/remote-home/share/lwang_share/models/AI-ModelScope/roberta-large/"

# Load RoBERTa tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Read input data
with open(input_file, 'r', encoding='utf-8') as f:
    data_list = json.load(f)

# Store filtered data
filtered_data = []

# Process each item
for item in data_list:
    # Extract rating from input text
    input_text = item.get("input", "")
    rating = None
    for r in ["1.0", "2.0", "3.0", "4.0", "5.0"]:
        if f"rating of \"{r}\"" in input_text:
            rating = r
            break
    if not rating:
        continue  # Skip if no rating found

    # Filter profile entries where 'overall' matches the rating
    profile = item.get("profile", [])
    filtered_profile = [
        entry for entry in profile
        if str(entry.get("overall", "")) == rating
    ]

    # Skip if no matching entries
    if not filtered_profile:
        continue

    # Sort by word count in reviewText (ascending)
    filtered_profile.sort(key=lambda x: len(x.get("reviewText", "").split()))

    # Further filter by token length (<= 250 tokens)
    valid_profile = []
    for entry in filtered_profile:
        review_text = entry.get("reviewText", "")
        tokens = tokenizer.encode(review_text, truncation=False, padding=False)
        if len(tokens) <= 250:
            valid_profile.append(entry)

    # Skip if all reviews are too long
    if not valid_profile:
        continue

    # Keep all other fields, only update profile
    new_item = {key: value for key, value in item.items() if key != "profile"}
    new_item["profile"] = valid_profile
    filtered_data.append(new_item)

# Save filtered data
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"Processing completed. {len(filtered_data)} items retained and saved to {output_file}")

random.seed(42)
sampled_data = random.sample(filtered_data, min(1200, len(filtered_data)))  # Avoid sampling more than available

# Save sampled data
sampled_output_file = output_file.replace(".json", "_sampled_1200.json")
with open(sampled_output_file, 'w', encoding='utf-8') as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=2)

print(f"Random sampling completed. {len(sampled_data)} items sampled (seed=42) and saved to {sampled_output_file}")