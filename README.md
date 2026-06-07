# 🔍 Hybrid ML Framework for Fake News Detection

> **BERT + XGBoost hybrid model** that combines transformer-based semantic embeddings with gradient-boosted ensemble classification to detect fake news articles.

---

## 📊 Results

| Model | Accuracy |
|---|---|
| NLP Model (BERT standalone) | 85% |
| Ensemble Model (XGBoost standalone) | 82% |
| **Hybrid Model (BERT + XGBoost)** | **88%** |

The hybrid approach outperforms both individual models — BERT captures deep linguistic context, XGBoost provides robust classification on top of those embeddings.

---

## 🧠 How It Works

### Architecture

```
News Article (raw text)
        │
        ▼
 BERT Tokenizer
 (bert-base-uncased)
        │
        ▼
 BERT Encoder → [CLS] token embedding (768-dim vector)
        │
        ▼
 XGBoost Classifier
 (trained on BERT embeddings)
        │
        ▼
 Prediction: REAL (1) or FAKE (0)
```

### Why BERT + XGBoost?

- **BERT** generates rich, context-aware sentence embeddings that capture semantic nuance, idiom, and tone — things TF-IDF can't.
- **XGBoost** is fast, interpretable, and resistant to overfitting. It handles high-dimensional input (768-dim embeddings) efficiently via gradient boosting.
- **Together**: BERT handles the linguistic complexity; XGBoost handles the decision boundary. The result is better recall and F1 on complex/ambiguous articles.

---

## 📁 Repository Structure

```
fake-news-detection/
│
├── fake_news_bert_xgb.py       # Main training + evaluation script
├── requirements.txt            # All dependencies
├── README.md                   # This file
│
├── data/
│   └── WELFake_Dataset.csv     # Dataset (see Data section below)
│
└── results/
    └── classification_report.txt   # Output metrics
```

---

## 📦 Dataset

This project uses the **WELFake Dataset** — a large-scale fake news dataset combining four public datasets (Kaggle, McIntire, Reuters, BuzzFeed Political):

- **72,134 articles** total
- **35,028 real** / **37,106 fake**
- Columns: `title`, `text`, `label` (1 = real, 0 = fake)

**Download:** [WELFake on Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

Place the CSV at `data/WELFake_Dataset.csv` before running.

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detection.git
cd fake-news-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Download `WELFake_Dataset.csv` from Kaggle (link above) and place it in `data/`.

### 4. Run training + evaluation

```bash
python fake_news_bert_xgb.py
```

This will:
- Load and preprocess the dataset
- Generate BERT embeddings for all articles (may take 10–30 min depending on hardware)
- Train XGBoost on the embeddings
- Print a full classification report

> **Note:** Running BERT on CPU is slow. Google Colab (free GPU) is recommended. The original notebook was run on Colab: [Open in Colab](https://colab.research.google.com/drive/1Yff1Xe0gGSOdbPv1dKYjeSGNlfs_ervH)

---

## 📋 Requirements

```
pandas
scikit-learn
transformers
torch
xgboost
numpy
```

Install all at once:
```bash
pip install pandas scikit-learn transformers torch xgboost numpy
```

---

## 📈 Methodology

### Data Preprocessing
- Dropped rows with missing `title`, `text`, or `label` fields
- Converted labels to numeric integers (0 = fake, 1 = real)
- 80/20 train-test split (stratified by label, `random_state=42`)

### Feature Extraction (BERT)
- Tokenized text using `bert-base-uncased` tokenizer (`max_length=512`, padded + truncated)
- Extracted the `[CLS]` token's embedding from the final hidden state — a 768-dimensional vector representing the full article's meaning
- No fine-tuning of BERT weights (frozen encoder, embeddings used as static features)

### Classification (XGBoost)
- `XGBClassifier` with 100 estimators trained on the 768-dim BERT embeddings
- Evaluated with precision, recall, F1-score, and accuracy

### Why not fine-tune BERT end-to-end?
Fine-tuning requires significantly more compute and GPU memory. Using BERT as a frozen feature extractor + XGBoost on top achieves strong results with far lower resource requirements — ideal for research settings without dedicated ML hardware.

---

## 🔮 Future Work

- [ ] Multimodal extension: incorporate article images alongside text
- [ ] Fine-tune BERT end-to-end for higher accuracy
- [ ] Real-time detection API (FastAPI wrapper)
- [ ] Multilingual evaluation (cross-lingual BERT variants)
- [ ] Model quantization / pruning for lightweight deployment

---

## 👥 Authors

- **Muhammad Hamdan** — [github.com/YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- **Aleena Ashfaq**

---

## 📄 References

1. Berrondo-Otermin & Sarasa-Cabezuelo — AI techniques in fake news detection
2. Shu et al. — Social media misinformation: patterns, user credibility, propagation
3. Agarwal & Dixit — Ensemble learning for detection robustness
4. Springer — Transformer-based multilingual detection
5. Wang — LIAR dataset: labeled truthful/deceptive statements
6. Ahmed et al. — Text classification for opinion spam and fake news detection
