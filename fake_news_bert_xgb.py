# -*- coding: utf-8 -*-
"""
Hybrid Fake News Detection: BERT Embeddings + XGBoost Classifier

Architecture:
  Raw text → BERT tokenizer → [CLS] embedding (768-dim) → XGBoost → REAL / FAKE

Dataset: WELFake_Dataset.csv
  Download: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
  Place at: data/WELFake_Dataset.csv
"""

import os
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH   = "data/WELFake_Dataset.csv"
MAX_LENGTH  = 512
TEST_SIZE   = 0.2
RANDOM_SEED = 42
XGB_ESTIMATORS = 100
BATCH_SIZE  = 32   # encode BERT in batches to avoid OOM on large datasets

# ── 1. Load & clean dataset ─────────────────────────────────────────────────
print("Loading dataset...")
data = pd.read_csv(DATA_PATH, encoding="utf-8", on_bad_lines="skip", quoting=3)
data = data[["title", "text", "label"]].dropna()
data["label"] = pd.to_numeric(data["label"], errors="coerce")
data = data.dropna(subset=["label"])

features = data["text"].reset_index(drop=True)
labels   = data["label"].astype(int).reset_index(drop=True)

print(f"  Total samples : {len(data)}")
print(f"  Real (1)      : {(labels == 1).sum()}")
print(f"  Fake (0)      : {(labels == 0).sum()}")

# ── 2. Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=labels
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ── 3. Load BERT ─────────────────────────────────────────────────────────────
print("\nLoading BERT model (bert-base-uncased)...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert      = BertModel.from_pretrained("bert-base-uncased")
bert.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert.to(device)
print(f"  Running on: {device}")

# ── 4. BERT embedding extraction ─────────────────────────────────────────────
def get_bert_embeddings(texts: pd.Series) -> np.ndarray:
    """
    Extract [CLS] token embeddings from BERT for a series of texts.
    Processes in batches to avoid memory issues.

    Returns: ndarray of shape (n_samples, 768)
    """
    all_embeddings = []
    texts_list = texts.tolist()

    for start in range(0, len(texts_list), BATCH_SIZE):
        batch = texts_list[start : start + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = bert(**inputs)

        # [CLS] token is the first token; its embedding represents the whole sequence
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)

        if (start // BATCH_SIZE) % 5 == 0:
            print(f"  Encoded {min(start + BATCH_SIZE, len(texts_list))}/{len(texts_list)} samples...")

    return np.vstack(all_embeddings)


print("\nGenerating BERT embeddings for training set...")
X_train_emb = get_bert_embeddings(X_train)

print("\nGenerating BERT embeddings for test set...")
X_test_emb = get_bert_embeddings(X_test)

print(f"\nEmbedding shape: {X_train_emb.shape}")  # (n_samples, 768)

# ── 5. Train XGBoost ──────────────────────────────────────────────────────────
print("\nTraining XGBoost classifier...")
clf = xgb.XGBClassifier(
    n_estimators=XGB_ESTIMATORS,
    random_state=RANDOM_SEED,
    use_label_encoder=False,
    eval_metric="logloss",
)
clf.fit(X_train_emb, y_train)
print("  Training complete.")

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
print("\nEvaluating on test set...")
predictions = clf.predict(X_test_emb)

accuracy = accuracy_score(y_test, predictions)
report   = classification_report(y_test, predictions, target_names=["Fake (0)", "Real (1)"])

print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nClassification Report:")
print(report)

# ── 7. Save results ───────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
with open("results/classification_report.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

print("Results saved to results/classification_report.txt")
