### Sprint 1: Data Pipelines & Candidate Generation (Week 1–2)
* Parse SPIDER v2 into a unified schema per entity domain (e.g., handling missing fields, text normalization).
* Implement a fast blocking strategy (BM25 or Locality Sensitive Hashing - LSH) to restrict candidate pairs from $O(N^2)$ to a manageable subset while preserving high recall ($>95\%$).

### Sprint 2: Lightweight Baseline (Modules A & B) (Week 3–4)
* **Module A Baseline:** 1D Character/Token CNN or lightweight TF-IDF projection.
* **Module B Baseline:** Standard Multi-Layer Perceptron (MLP) taking concatenated pair embeddings $(e_A, e_B, \vert{}e_A - e_B\vert{}, e_A \odot e_B)$.
* Define evaluation metrics early: **Pairwise Precision, Recall, F1**, and **Inference Time (ms/pair)**.

### Sprint 3: Deep Models & Graph Clustering (Week 5–6)
* **Module A Variations:** Fine-tune a Siamese Bi-Encoder (e.g., `all-MiniLM-L6-v2`).
* **Module B Variations:** Implement Cross-Encoder (DistilBERT) and compare with MLP.
* **Module C Variations:** 
  1. Connected Components with similarity thresholds.
  2. Graph Attention Network (GAT) trained on pair confidence scores to prune transitive inconsistencies.

### Sprint 4: Full Modular Pipeline & Benchmarking (Week 7–8)
* Connect all module combinations into an automated benchmarking script.
* Measure trade-offs: Accuracy (F1, Adjusted Rand Index) vs. Computational Efficiency (Memory usage, Latency, FLOPs).

### Sprint 5: Paper Drafting & Results Visualization (Week 9–10)
* Create figures for ablation studies and confusion matrices.
* Write paper sections using the structure outlined below.