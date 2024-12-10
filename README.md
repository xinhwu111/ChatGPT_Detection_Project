# AI-Generated Content Detection with RoBERTa

This repository contains the implementation of a project aimed at distinguishing between human-generated and AI-generated content using a fine-tuned RoBERTa-based binary classifier. The project leverages the HC3 (Human ChatGPT Comparison Corpus) dataset and provides insights into linguistic and stylistic differences between human and ChatGPT responses.

## Overview

The rapid rise of large language models (LLMs) like ChatGPT has revolutionized natural language processing (NLP). While these models excel in generating coherent and contextually accurate responses, they pose risks such as misinformation and ethical misuse. This project develops a robust system to identify AI-generated content, contributing to transparency and responsible AI usage.

### Key Objectives:
- Analyze linguistic differences between human and ChatGPT-generated responses.
- Train and evaluate a RoBERTa-based binary classifier.
- Provide interpretability through token-level predictability and other visual analyses.

## Dataset

The project uses the HC3 dataset, which includes:
- **24,322 samples** spanning multiple domains such as finance, medicine, and open-domain QA.
- Each sample contains a question, a human-generated response, and a ChatGPT-generated response.

The dataset ensures cross-domain robustness, making it suitable for evaluating real-world scenarios.

## Methodology

### Problem Formulation
The task is framed as a binary classification problem:
- **Input**: A textual response (human or AI-generated).
- **Output**: A binary label (\`0\` for human-generated, \`1\` for AI-generated).

### Model Training
- **Model**: RoBERTa-base with a binary classification head.
- **Framework**: Hugging Face Transformers.
- **Hyperparameters**:
  - Learning rate: `1e-5`
  - Epochs: `3`
  - Batch size: `4`
  - Weight decay: `0.01`
- Optimizer: AdamW
- Evaluation: Accuracy, precision, recall, and F1-score.

### Exploratory Data Analysis (EDA)
EDA visualizations include:
- Source distribution of the dataset.
- Comparison of average response lengths.
- Vocabulary density and part-of-speech (POS) distributions.
- Sentiment analysis results.

### Evaluation
- Confusion matrix and classification metrics confirm near-perfect performance with:
  - Precision: `1.00`
  - Recall: `1.00`
  - F1-score: `1.00`.

## Results

Key findings include:
- **Human Responses**: High variability in length, vocabulary density, and POS usage.
- **ChatGPT Responses**: Consistent, predictable, and structured outputs.
- **Model Performance**: The fine-tuned RoBERTa model achieves exceptional accuracy, effectively distinguishing human from AI-generated text.

## Visualizations
The repository includes visualizations of:
- EDA results (e.g., sentiment analysis, response lengths).
- Training and validation metrics (e.g., loss curves, accuracy).
- Token-level predictability highlighting for both human and ChatGPT-generated text.

## Conclusion

This project demonstrates the effectiveness of RoBERTa in detecting AI-generated content. The findings provide valuable insights into the stylistic and structural differences between human and machine-generated text, highlighting the potential for transformer-based models in responsible AI deployment.

## Installation and Usage

### Requirements
- Python 3.7+
- Libraries:
  - `transformers`
  - `datasets`
  - `torch`
  - `pandas`
  - `matplotlib`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repository-name>.git
