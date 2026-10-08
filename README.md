# Deep Entity Resolution

A modular deep learning pipeline for **multi-field entity resolution** on structured and semi-structured tabular data.

The project implements an end-to-end entity resolution workflow combining **candidate generation, field-level representation learning, pairwise matching, and graph-based entity clustering**. The system is designed to support comparative experiments across different blocking, matching, and clustering approaches.

The primary benchmark is **FEBRL-2**, a synthetic entity resolution dataset containing 5,000 records representing approximately 4,000 underlying entities, including noisy duplicate records and multiple records belonging to the same entity.

> This project was developed as the final project for a Deep Learning course.

---

## Project Overview

Entity resolution is the task of determining which records refer to the same real-world entity despite missing values, formatting differences, typographical errors, and other forms of noise.

This project implements a modular pipeline:

```text
Raw Records
     │
     ▼
Preprocessing
     │
     ▼
Blocking / Candidate Generation
     │
     ▼
Field-level Feature Extraction
     │
     ▼
Pairwise Deep Matching
     │
     ▼
Similarity / Match Scores
     │
     ▼
Graph Construction
     │
     ▼
Entity Clustering
```

The modular design allows different approaches to be evaluated while keeping the overall entity resolution pipeline consistent.

---

## Architecture & Component Design

| Pipeline Step                   | Module                | Description                                                                 | Implemented / Planned Variations                  |
| ------------------------------- | --------------------- | --------------------------------------------------------------------------- | ------------------------------------------------- |
| **0. Input Data**               | Dataset               | Multi-field person records containing noisy and duplicate entries           | **FEBRL-2**                                       |
| **1. Preprocessing & Blocking** | Candidate Generation  | Efficiently reduces the number of record pairs that require deep comparison | Rule-based blocking, LSH                          |
| **2. Feature Extraction**       | Field Representations | Learns representations for individual record fields                         | **CharCNN**, Siamese/Bi-Encoder                   |
| **3. Deep Matcher**             | Pair Classification   | Predicts whether two records refer to the same entity                       | **CNN + MLP**, Cross-Encoder, ResMLP              |
| **4. Graph Clustering**         | Entity Resolution     | Converts pairwise predictions into entity-level clusters                    | Connected Components, Graph-based refinement, GAT |

The main baseline model uses a **field-wise Character CNN followed by an MLP pair matcher**. Each field is encoded independently before the representations of the two records are combined for pairwise classification.

---

## Dataset

The project uses **FEBRL-2** from the Febrl record linkage benchmark.

The dataset contains:

* **5,000 total records**
* Approximately **4,000 underlying entities**
* **1,000 duplicate records**
* Up to **5 duplicate records for a single original entity**
* 10 entity attributes:

```text
given_name
surname
street_number
address_1
address_2
suburb
postcode
state
date_of_birth
soc_sec_id
```

Unlike a simple one-to-one duplicate dataset, FEBRL-2 contains entities represented by multiple noisy records. This makes it suitable for evaluating both **pairwise entity matching** and subsequent **graph-based clustering**.

Ground-truth linkage information is used to identify records belonging to the same entity.

---

## Blocking & Candidate Generation

Comparing every possible pair of records has quadratic complexity: $O(n²)$

For this reason, the pipeline first generates a smaller set of plausible candidate pairs through blocking.

Current blocking approaches include:

* Rule-based blocking
* Locality-Sensitive Hashing (LSH)
* Multiple blocking keys
* Candidate-pair union across blocking strategies

The blocking stage is evaluated separately using:

* Candidate reduction ratio
* Pair completeness / blocking recall
* Candidate set size

The goal is to reduce the number of comparisons while retaining the majority of true entity matches.

---

## Deep Matching

### Baseline: Field-wise Character CNN + MLP

The primary matching architecture uses a separate Character CNN encoder for each entity field.

Conceptually:

```text
                 Record A
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
   Given Name    Surname       Address ...
       │            │            │
     CharCNN      CharCNN      CharCNN
       │            │            │
       └────────────┼────────────┘
                    │
                    ▼
              Record A Embedding


                 Record B
                    │
                    ▼
            Field-wise CharCNN
                    │
                    ▼
              Record B Embedding

                    │
                    ▼
          Pair Feature Construction
                    │
                    ▼
                MLP Matcher
                    │
                    ▼
             Match Probability
```

The model learns field-level representations while allowing the MLP to learn interactions between the two records.

The output is a pairwise match score indicating whether the two records are likely to represent the same entity.

---

## Planned Model Comparisons

The modular architecture allows the baseline to be compared with deeper alternatives.

### Feature Representation

**V1 — Character CNN**

Field-specific character-level convolutional representations.

**V2 — Siamese / Bi-Encoder**

Transformer-based field or record representations using a shared encoder.

**V3 — Hybrid Features**

Combination of learned representations with traditional string similarity features.

### Pairwise Matcher

**V1 — MLP**

Dense neural classifier operating on the pair representations.

**V2 — Cross-Encoder**

Jointly encodes the two records to model cross-record interactions.

**V3 — Residual MLP**

Deeper MLP architecture with residual connections.

---

## Graph-Based Entity Resolution

Pairwise predictions are converted into a graph:

```text
Record A ───── Record B
   │              │
   │              │
   └──── Record C ┘
```

where:

* Nodes represent records.
* Edges represent candidate record pairs.
* Edge weights represent predicted match confidence.

The graph can then be used to infer entity-level clusters.

### Current / Planned Approaches

* **Connected Components**

  * Apply a confidence threshold to predicted edges.
  * Assign connected records to the same entity.

* **Graph-based Refinement**

  * Use graph structure to identify inconsistent pairwise predictions.

* **Graph Attention Network (GAT)**

  * Learn from local graph structure to refine pairwise confidence and reduce transitive inconsistencies.

This stage is particularly relevant for FEBRL-2 because individual entities may have multiple duplicate records.

---

## Evaluation

The pipeline evaluates performance at multiple levels.

### Blocking

* Candidate pair count
* Reduction ratio
* Blocking recall / pair completeness

### Pairwise Matching

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC

### Entity-Level Resolution

* Cluster precision / recall
* Cluster F1
* Adjusted Rand Index (ARI)

The evaluation separates **pairwise matching performance** from **final entity clustering performance**.

---

## Repository Structure

```text
Deep_Entity_Resolution/
│
├── data/
│   ├── raw/                  # Raw FEBRL datasets
│   ├── processed/            # Cleaned / transformed datasets
│   └── pairs/                # Candidate pairs generated by blocking
│
├── src/
│   ├── etl.py                # Data loading and preprocessing
│   ├── blocking.py           # Blocking and candidate generation
│   ├── feature_extractor.py  # CNN / Bi-Encoder feature extraction
│   ├── classifiers.py        # Pairwise matching models
│   ├── clustering.py         # Graph construction and clustering
│   ├── utils/
│   │   └── metrics.py        # Evaluation metrics and utilities
│   └── train.py              # Training and evaluation pipeline
│
├── configs/                  # Experiment configuration files
│
├── notebooks/                # EDA and experiment notebooks
│
├── benchmarks/               # Results, comparisons, and ablation studies
│
├── requirements.txt
│
└── README.md
```

---

## Experimental Strategy

The project evaluates the entity resolution pipeline incrementally:

```text
Blocking
   │
   ▼
Candidate Generation
   │
   ▼
Rule-based Baseline
   │
   ▼
CNN + MLP
   │
   ├── Model comparisons
   │
   ▼
Pairwise Similarity Graph
   │
   ▼
Connected Components
   │
   ▼
Graph Refinement
```

Experiments focus on understanding the trade-off between:

* Candidate reduction
* Pairwise matching accuracy
* Computational cost
* Entity-level clustering quality

---

## Project Status

### Completed

* [x] FEBRL dataset integration
* [x] Data inspection and preprocessing
* [x] Multi-field pair representation
* [x] Character-level CNN feature extraction
* [x] MLP pair matcher
* [x] Training and validation pipeline
* [x] Blocking evaluation
* [x] Pairwise evaluation metrics

### In Progress

* [ ] FEBRL-2 hard-negative evaluation
* [ ] Improved blocking strategies
* [ ] Siamese / Bi-Encoder comparison
* [ ] Cross-Encoder comparison
* [ ] Graph-based clustering
* [ ] GAT-based refinement
* [ ] Ablation studies

---

## Goal

The objective is not only to determine whether two individual records match, but to build a complete and modular **entity resolution system** capable of progressing from noisy raw records to consistent entity-level clusters.

The project therefore evaluates both:

```text
Pairwise Matching
        +
Global Entity Resolution
```

rather than treating entity resolution as a standalone binary classification problem.
