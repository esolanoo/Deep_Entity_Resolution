# Deep_Entity_Resolution
A modular, lightweight Deep Learning pipeline for Multi-Field Entity Resolution on structured and semi-structured tabular datasets

An end-to-end, modular Deep Learning pipeline for Multi-Field Entity Resolution on complex tabular/relational datasets (benchmarked on [SPIDER v2](https://figshare.com/articles/dataset/SPIDER_v2_Synthetic_Person_Information_Dataset_for_Entity_Resolution/30472712)).

## Project Overview
This repository implements a modular, comparative benchmark for deep entity resolution. The pipeline standardizes candidate pair generation, multi-field feature extraction, neural match classification, and global graph clustering into four interchangeable modules.

---

## Architecture & Component Design

| Pipeline Step | Module Description | Implemented Variations |
| :--- | :--- | :--- |
| **0. Input Data** | Multi-field raw records | SPIDER v2 subsets |
| **1. Preprocessing & Blocking** | Indexing & candidate pair generation | Token-based LSH / BM25 / TF-IDF Blocking |
| **2. Feature Extractor** | Field-wise & cross-field embeddings | • **V1:** Light CNN (1D-Char/Token CNN)<br>• **V2:** Siamese Bi-Encoder (MiniLM / RoBERTa-Small)<br>• **V3:** Hybrid String Metric + Feature Projection |
| **3. Deep Matcher** | Pairwise similarity classification | • **V1:** MLP with Field Interaction Layers<br>• **V2:** Cross-Encoder (DistilBERT)<br>• **V3:** Dense Residual Network (ResNet-style MLP) |
| **4. Graph Clustering** | Entity resolution & ID assignment | • **V1:** Hardcoded Threshold (Connected Components)<br>• **V2:** Correlation Clustering / Markov Clustering<br>• **V3:** Graph Attention Network (GAT) for Link Prediction/Refinement |

---

## Repository Structure

```text
├── data/
│   ├── raw/                 # SPIDER v2 raw tables
│   ├── processed/           # Tokenized & cleaned datasets
│   └── pairs/               # Candidate pairs generated from blocking
├── src/
│   ├── blocking/            # Blocking & Candidate Pair generation
│   ├── feature_extractors/  # CNN, Bi-Encoder, Hybrid feature extractors
│   ├── classifiers/         # MLP, Cross-Encoder, ResMLP models
│   ├── clustering/          # Connected components, GAT, Graph clustering
│   ├── utils/               # Metrics (Pairwise F1, Clustering F1/ARI), logging
│   └── train.py             # Main training loop
├── configs/                 # YAML configuration files for experiments
├── notebooks/               # EDA and performance visualization
├── benchmarks/              # Results, metric comparisons, and ablation logs
├── requirements.txt
└── README.md
```


Chinnappa, Praveen; Arokiya Dass, Rose Mary; mathur, yash (2025). SPIDER (v2): Synthetic Person Information Dataset for Entity Resolution. figshare. Dataset. https://doi.org/10.6084/m9.figshare.30472712.v2
