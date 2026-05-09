# News Article Summarization using NLP

A deep learning–based text summarization project that generates concise summaries for lengthy news articles using Sequence-to-Sequence (Seq2Seq) models with attention mechanisms.

This project compares two popular attention techniques:

* Bahdanau Attention
* Luong Attention

The models are implemented using TensorFlow and trained on a news summarization dataset.

---

# Features

* Automatic news article summarization
* Seq2Seq architecture using LSTM networks
* Comparison of:

  * Bahdanau Attention
  * Luong Attention
* Text preprocessing and cleaning pipeline
* Tokenization and sequence padding
* Early stopping to reduce overfitting
* Model evaluation using ROUGE metrics
* Training visualization using Matplotlib

---

# Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Transformers
* ROUGE Score
* NLP Techniques

---

# Project Workflow

1. Install dependencies
2. Load news dataset
3. Clean and preprocess text
4. Tokenize articles and summaries
5. Build Seq2Seq models
6. Apply attention mechanisms
7. Train models
8. Evaluate summarization performance
9. Compare attention mechanisms

---

# Installation

## Clone Repository

```bash
git clone <your-repository-url>
cd <repository-folder>
```

## Install Dependencies

```bash
pip install tensorflow pandas numpy matplotlib rouge_score transformers sentencepiece
```

---

# Dataset

The project expects the following CSV files:

```text
train.csv
validation.csv
test.csv
```

Required columns:

* `article` → Full news article
* `highlights` → Target summary

---

# Configuration

The notebook uses the following default configuration:

```python
max_vocab = 50000
max_article_len = 400
max_summary_len = 50
embedding_dim = 128
latent_dim = 128
epochs = 20
batch_size = 64
```

---

# Text Preprocessing

The preprocessing pipeline includes:

* Lowercasing
* Removing special characters
* Removing extra spaces
* Tokenization
* Sequence padding

Example cleaning function:

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text
```

---

# Model Architecture

The project uses a Seq2Seq Encoder–Decoder architecture with:

* Embedding Layer
* LSTM Encoder
* LSTM Decoder
* Attention Layer
* Dense Output Layer

---

# Attention Mechanisms

## Bahdanau Attention

Additive attention mechanism that learns alignment scores between encoder and decoder states.

## Luong Attention

Multiplicative attention mechanism using dot-product scoring for faster computation.

---

# Model Training

## Train Bahdanau Attention Model

```python
bahdanau_model = build_model('bahdanau')
```

## Train Luong Attention Model

```python
luong_model = build_model('luong')
```

Early stopping is used to improve generalization and prevent overfitting.

---

# Evaluation

The project evaluates generated summaries using:

* ROUGE-1
* ROUGE-2
* ROUGE-L

These metrics measure overlap between generated and reference summaries.

---

# Visualizations

Training graphs include:

* Training Loss
* Validation Loss
* Performance comparison between attention mechanisms

Visualization is implemented using Matplotlib.

---

# Output Example

## Input Article

```text
Long news article content...
```

## Generated Summary

```text
Short summarized version of article.
```

---

# Applications

* News aggregation platforms
* Content summarization systems
* Information retrieval systems
* AI-powered news assistants
* Text analytics platforms

---

# Future Improvements

* Transformer-based summarization
* BERT/T5 integration
* Beam search decoding
* Real-time summarization API
* Multilingual summarization
* Web application deployment

---

# Results

The project demonstrates effective abstractive summarization using attention-based Seq2Seq models and compares the performance of Bahdanau and Luong attention mechanisms.

---

# License

This project is intended for educational and research purposes.

You may modify and extend it according to your requirements.

---

# Author

Developed as an NLP-based News Article Summarization project using TensorFlow and attention-based Seq2Seq architectures.
