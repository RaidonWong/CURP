import torch
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
import os
import tqdm
import random

# ==================== Configuration ====================
roberta_path = "/remote-home/share/lwang_share/models/AI-ModelScope/roberta-large/"
excel_path = "/remote-home/share/lwang_share/understand_generate/codebook/Codebook_modified.xlsx"
sheet_name = "Sheet12"
output_path = "/remote-home/share/lwang_share/understand_generate/codebook/stage2_450.pt"

# Target size for codebook
TARGET_SIZE = 270

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ======================================================
# Step 1: Load text data from Excel
# ======================================================
print(" Loading text data from Excel...")
try:
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
except Exception as e:
    raise FileNotFoundError(f"Failed to load Excel file: {e}")

# Extract first column, drop NaN, and convert to string
texts = df.iloc[:, 0].dropna().astype(str).tolist()
print(f" Loaded {len(texts)} raw text entries.")

# ======================================================
# Step 2: Sample or upsample to target size
# ======================================================
print(f" Target codebook size: {TARGET_SIZE}")

if len(texts) >= TARGET_SIZE:
    print(f"  Sampling {TARGET_SIZE} unique entries without replacement...")
    sampled_texts = random.sample(texts, TARGET_SIZE)
else:
    print(f"  Insufficient data ({len(texts)} < {TARGET_SIZE}). Sampling with replacement...")
    sampled_texts = random.choices(texts, k=TARGET_SIZE)

print(f" Final number of texts: {len(sampled_texts)}")

# ======================================================
# Step 3: Load RoBERTa model and encode texts
# =================================================-----
print(" Loading RoBERTa model for encoding...")
tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
model = RobertaModel.from_pretrained(roberta_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(" Encoding texts into [CLS] embeddings...")
embeddings = []

with torch.no_grad():
    for i, text in enumerate(tqdm(sampled_texts, desc="Encoding texts")):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=20
        ).to(device)

        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (1, hidden_size)
        embeddings.append(cls_embedding.cpu())

# ======================================================
# Step 4: Combine and save embedding matrix
# ======================================================
embedding_matrix = torch.cat(embeddings, dim=0)
print(f" Embedding matrix shape: {embedding_matrix.shape}")

os.makedirs(os.path.dirname(output_path), exist_ok=True)
torch.save(embedding_matrix, output_path)
print(f" Codebook embeddings saved to: {output_path}")