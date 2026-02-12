# N-Gram Language Modeling & Domain Shift Analysis

## 📌 Project Overview
This project implements **N-gram Language Models (Unigram, Bigram, Trigram)** from scratch for **CS60075 Natural Language Processing (Assignment 1)**[cite: 1].

The goal is to predict the next word in a sequence based on probability estimates derived from a training corpus[cite: 16]. The models are trained on Wikipedia articles and evaluated on three different domains to analyze the impact of **domain shift** on model performance[cite: 41].

Key features include:
* **Maximum Likelihood Estimation (MLE)** for probability calculation[cite: 51].
* **Smoothing Techniques**: Implementation of Laplace (Add-1)  and Good-Turing Discounting [cite: 65] to handle unseen n-grams.
* **Perplexity Evaluation**: Comparing model performance across In-Domain (Wikipedia) and Out-of-Domain (Scientific, Financial) datasets[cite: 104, 105].

---

## 📂 Repository Structure

The codebase is modularized for efficiency and readability:

```text
├── data/                   # Dataset folder (Place CSVs here)
│   ├── wiki_train.csv      # Training Corpus (~14k articles) [cite: 38]
│   ├── wiki_test.csv       # In-domain test set [cite: 39]
│   ├── arxiv_test.csv      # Scientific domain test set [cite: 39]
│   └── financial_test.csv  # Financial domain test set [cite: 39]
├── main.py                 # Entry point: Orchestrates training and evaluation
├── models.py               # Logic for N-gram counting, Laplace, and Good-Turing smoothing
├── preprocessing.py        # Text cleaning, tokenization, and <UNK> replacement
├── evaluation.py           # Perplexity calculation and report generation
└── README.md               # Project documentation


Gemini said
Here is the complete, single-file README.md content. You can copy this entirely and paste it into your repository.

Markdown
# N-Gram Language Modeling & Domain Shift Analysis

## 📌 Project Overview
This project implements **N-gram Language Models (Unigram, Bigram, Trigram)** from scratch for **CS60075 Natural Language Processing (Assignment 1)**[cite: 1].

The goal is to predict the next word in a sequence based on probability estimates derived from a training corpus[cite: 16]. The models are trained on Wikipedia articles and evaluated on three different domains to analyze the impact of **domain shift** on model performance[cite: 41].

Key features include:
* **Maximum Likelihood Estimation (MLE)** for probability calculation[cite: 51].
* **Smoothing Techniques**: Implementation of Laplace (Add-1)  and Good-Turing Discounting [cite: 65] to handle unseen n-grams.
* **Perplexity Evaluation**: Comparing model performance across In-Domain (Wikipedia) and Out-of-Domain (Scientific, Financial) datasets[cite: 104, 105].

---

## 📂 Repository Structure

The codebase is modularized for efficiency and readability:

```text
├── data/                   # Dataset folder (Place CSVs here)
│   ├── wiki_train.csv      # Training Corpus (~14k articles) [cite: 38]
│   ├── wiki_test.csv       # In-domain test set [cite: 39]
│   ├── arxiv_test.csv      # Scientific domain test set [cite: 39]
│   └── financial_test.csv  # Financial domain test set [cite: 39]
├── main.py                 # Entry point: Orchestrates training and evaluation
├── models.py               # Logic for N-gram counting, Laplace, and Good-Turing smoothing
├── preprocessing.py        # Text cleaning, tokenization, and <UNK> replacement
├── evaluation.py           # Perplexity calculation and report generation
└── README.md               # Project documentation

# ⚙️ Installation & Requirements
Ensure you have Python 3.x installed. The project relies on standard NLP and data libraries.   
Clone the repository:
Bash
git clone [https://github.com/Adityaraj142857/NLP.git](https://github.com/Adityaraj142857/NLP.git)
cd NLP
Install dependencies:
Bash
pip install pandas numpy nltk tqdm
NLTK Data: The script automatically checks for and downloads required NLTK resources (stopwords, wordnet, punkt) if they are missing.

