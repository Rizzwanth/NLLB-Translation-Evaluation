# NLLB Translation Evaluation

## Overview
This project implements a comprehensive machine translation evaluation methodology, prioritizing advanced sequence-to-sequence translation over traditional classification techniques. It utilizes Meta's `nllb-200-distilled-1.3B` model to translate and evaluate text across multiple languages (including English, Telugu, and Spanish) against the official FLORES-200 dataset benchmark. 

## Key Features
* **NLLB-200 Integration:** Leverages the distilled 1.3B parameter NLLB model for high-quality cross-lingual translation tasks.
* **FLORES-200 Benchmark:** Employs the `devtest` split of the FLORES-200 dataset for robust, ground-truth evaluation.
* **Advanced Metrics:** Calculates industry-standard metrics, including SacreBLEU and chrF++ (which is specifically advantageous for morphologically rich languages like Telugu).
* **Cross-Lingual Embeddings:** Extracts and visualizes text embeddings across different languages using UMAP for dimensionality reduction.
* **Kaggle-Optimized Pipeline:** Features custom Python data loading procedures configured to handle environment-specific file paths and sandbox constraints seamlessly within Kaggle.

## Tech Stack
* **Frameworks:** PyTorch, Hugging Face Transformers, Datasets
* **Evaluation:** SacreBLEU
* **Visualization:** Matplotlib, Seaborn, UMAP-learn
* **Environment:** Designed for GPU-accelerated execution (e.g., Kaggle Tesla T4)

## Installation & Usage
To run this notebook, ensure you have the required dependencies installed:

```bash
pip install transformers datasets sacrebleu torch matplotlib seaborn umap-learn "numpy<2.0"
