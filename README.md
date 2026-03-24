
# 📝 CSL 7640: Natural Language Understanding - Assignment 2

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-Processing-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

This repository contains the complete source code, datasets, and final technical report for **Assignment 2 of the CSL 7640: Natural Language Understanding**. 

The project tackles two core NLP challenges: building domain-specific word embeddings from scratch and developing character-level Recurrent Neural Networks (RNNs) for text generation.

## 📑 Table of Contents
- [Problem 1: Word Embeddings (Word2Vec)](#-problem-1-word-embeddings-word2vec)
- [Problem 2: Character-Level RNN Generation](#-problem-2-character-level-rnn-generation)
- [Repository Structure](#-repository-structure)
- [Setup and Execution](#-setup-and-execution)
- [Final Report](#-final-report)

---

## 🧠 Problem 1: Word Embeddings (Word2Vec)
The goal of this task was to learn semantic word representations from a highly specialized, custom-built corpus containing official IIT Jodhpur academic documents (B.Tech curriculums, PG regulations, and research pages).

**Methodology:**
* **Corpus Prep:** Extracted text from PDFs using `PyPDF2` and web scraping. Cleaned and tokenized into ~23,500 words using `nltk`.
* **Models Implemented:** Continuous Bag of Words (CBOW) and Skip-gram with Negative Sampling (built from scratch in PyTorch).
* **Hyperparameters:** 50-dimensional embeddings, window size of 2, 5 negative samples.

**🌟 Key Findings:**
The models successfully learned deep semantic relationships directly from the university's data. 
> **Notable Analogy Discovered (CBOW):** `student : study :: faculty : assessment`
> *The model correctly mapped the functional roles within the university structure!*

---

## 🤖 Problem 2: Character-Level RNN Generation
This task focused on autoregressive sequence modeling. The objective was to train neural networks to generate novel, phonetically plausible Indian names character-by-character based on a dataset of 1,000 LLM-generated names.

**Models Compared:**
1. **Vanilla RNN:** Achieved the most realistic results (e.g., *"Ranya"*, *"Payindra"*).
2. **Bidirectional LSTM (BLSTM):** Suffered from severe overfitting and struggled with autoregressive generation without future context.
3. **Attention RNN:** Achieved high diversity but occasionally entered "stuttering" loops (e.g., *"Ananan"*).

**📊 Performance Metrics:**
| Model | Trainable Parameters | Novelty Rate | Diversity Rate |
| :--- | :--- | :--- | :--- |
| Vanilla RNN | 8,988 | 100.00% | 93.00% |
| BiLSTM | 54,684 | 100.00% | 88.00% |
| Attention RNN | 9,053 | 100.00% | 96.00% |

---

## 📂 Repository Structure

```text
├── data/
│   ├── raw/               # Original IIT Jodhpur PDFs used for the corpus
│   └── processed/         # Cleaned corpus and training name datasets
├── src/
│   ├── problem1_word2vec/ # Scripts for Word2Vec training, extraction, and PCA analysis
│   └── problem2_rnn/      # Scripts for RNN architectures, training, and evaluation
├── outputs/               # Saved model weights (.pt files) and visualization plots
├── Report.pdf             # Comprehensive technical report and qualitative analysis
└── README.md              # You are here!



⚙️ Setup and Execution
1. Clone the repository and install dependencies:
Bash
git clone [https://github.com/Sachinkumartandon/NLU-Assignment-2.git](https://github.com/Sachinkumartandon/NLU-Assignment-2.git)
cd NLU-Assignment-2
pip install torch torchvision numpy matplotlib nltk PyPDF2


2. Word2Vec Execution (Problem 1):
Navigate to the problem1_word2vec directory to run data preprocessing or model training.
Bash
python src/problem1_word2vec/train.py


3. RNN Generation (Problem 2):
To train the models from scratch and generate new Indian names:
Bash
python src/problem2_rnn/train.py


To evaluate novelty, diversity, and parameter counts:
Bash
python src/problem2_rnn/evaluate.py




