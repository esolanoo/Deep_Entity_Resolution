# Deep Learning for Multi-Field Entity Resolution Under Varying Record Ambiguity

## Abstract

Briefly describe the problem of resolving noisy records that refer to the same real-world entity. Present a modular pipeline consisting of blocking, learned pairwise matching, and graph-based entity resolution.

The study investigates two questions:

1. Whether learned pairwise similarity improves entity resolution for noisy, multi-record entities, and whether graph-based refinement improves consistency of the resulting entity clusters.
2. How the benefit of increasingly complex matching models changes as the proportion of exact field agreement between matching records changes.

Experiments are conducted primarily on FEBRL-2, with SPIDER v2 used to evaluate how dataset difficulty affects the necessity and effectiveness of deeper matching models.

---

# 1. Introduction

## 1.1 Problem

Entity resolution requires determining which records correspond to the same underlying entity despite:

* missing values
* typographical errors
* formatting differences
* inconsistent representations
* multiple records for the same entity

Traditional similarity rules can perform well when matching records contain many exact agreements, but their effectiveness may decrease as records become noisier and more ambiguous.

## 1.2 Motivation

A complete entity resolution system has two distinct problems:

1. **Pairwise matching:** determine whether two records represent the same entity.
2. **Entity-level resolution:** combine pairwise evidence to assign records to consistent entity clusters.

A high-confidence pairwise classifier does not necessarily guarantee globally consistent clusters. Graph-based methods provide a mechanism for incorporating relationships among multiple records.

## 1.3 Research Questions

### RQ1 — Learned Matching and Graph Refinement

> **Can learned pairwise similarity improve entity resolution under noisy, multi-record entities, and can graph-based refinement resolve inconsistencies in pairwise predictions?**

This question is evaluated using FEBRL-2, which contains multiple records belonging to the same underlying entities.

The experiment compares:

* exact/rule-based similarity baselines
* learned pairwise matching
* threshold-based graph clustering
* graph-based refinement

The evaluation considers both pairwise and entity-level performance.

### RQ2 — When Is a Deeper Model Necessary?

> **How does the benefit of increasingly complex matching models change as the proportion of exact field agreement between matching records changes?**

This question compares FEBRL-2 with SPIDER v2.

The objective is not simply to identify the highest-performing model, but to determine **when additional model complexity provides meaningful benefit over simpler matching approaches**.

---

# 2. Related Work

Discuss:

* classical record linkage
* blocking / candidate generation
* string similarity
* neural entity matching
* Siamese / bi-encoder architectures
* cross-encoders
* graph-based entity resolution

The emphasis should be on the progression:

```text
Rule-based similarity
        ↓
Learned pairwise similarity
        ↓
Graph-based global resolution
```

---

# 3. Datasets

## 3.1 FEBRL-2

Describe:

* 5,000 records
* approximately 4,000 underlying entities
* 1,000 duplicate records
* multiple noisy records per entity
* up to five duplicate records for one original
* ten available fields

The `soc_sec_id` field is excluded from the experiments because it exhibited near-exact agreement across positive pairs and therefore provided disproportionately strong identity evidence. Retaining it risked making the matching task dependent on a near-unique identifier rather than requiring the model to integrate noisy multi-field evidence

Report:

* missing-value rates
* number of entities
* number of records per entity
* distribution of exact field agreement among positive pairs
* distribution of exact field agreement among negative pairs

## 3.2 SPIDER v2

Describe:

* dataset size and structure
* field composition
* ground-truth construction
* exact-match distribution

SPIDER v2 is used primarily to study how model performance changes with dataset difficulty and exact-field agreement.

---

# 4. Methodology

## 4.1 Preprocessing

Normalize fields while preserving field-specific information.

Handle:

* missing values
* character/token encoding
* field-specific sequence lengths

## 4.2 Blocking

Blocking reduces the quadratic record-pair search space.

Evaluate:

* rule-based blocking
* LSH blocking
* union of blocking strategies

Metrics:

* candidate count
* reduction ratio
* blocking recall / pair completeness

## 4.3 Pairwise Matching

### Baseline

Exact field agreement / traditional similarity.

### Model 1 — CNN + MLP

Each field is independently encoded using a Character CNN.

The resulting field representations from the two records are combined and passed to an MLP classifier.

### Model 2 — Bi-Encoder

A shared neural encoder independently represents the two records before similarity computation.

### Model 3 — Cross-Encoder

The two records are jointly processed to model cross-record interactions.

Only models actually implemented should appear in the final experimental table.

---

# 5. RQ1 — Learned Pairwise Matching and Graph Resolution

## 5.1 Experimental Setup

Use FEBRL-2 with entity-disjoint train/validation/test partitions.

Positive pairs are records belonging to the same ground-truth entity.

Negative pairs are records belonging to different entities.

Evaluate both:

* random/easy negatives
* hard negatives with high field agreement

## 5.2 Pairwise Matching Results

Compare:

| Method               | Precision | Recall | F1 | PR-AUC | ROC-AUC |
| -------------------- | --------: | -----: | -: | -----: | ------: |
| Exact-field baseline |           |        |    |        |         |
| CNN + MLP            |           |        |    |        |         |
| Bi-Encoder           |           |        |    |        |         |
| Cross-Encoder        |           |        |    |        |         |

## 5.3 Graph Construction

Construct a graph where:

* nodes = records
* edges = predicted matching pairs
* edge weights = model confidence

## 5.4 Graph Resolution

Compare:

1. Ground-truth pair scores
2. Predicted pair scores
3. Thresholded Connected Components
4. Graph refinement / GAT

Evaluate:

* cluster precision
* cluster recall
* cluster F1
* ARI

## 5.5 Pairwise vs. Global Consistency

Analyze cases where pairwise predictions are individually plausible but collectively inconsistent.

Example:

```text
A ── 0.95 ── B
│
0.91
│
C

B ── 0.42 ── C
```

Determine whether graph structure improves the final entity assignment.

---

# 6. RQ2 — Model Necessity Under Different Dataset Difficulty

The goal of this experiment is to determine when a simple matching approach is sufficient and when deeper models provide measurable benefit.

For each dataset, characterize the positive-pair exact-match distribution.

For example:

```text
Low ambiguity
    ↓
Exact/rule-based methods already perform well
    ↓
CNN provides limited additional benefit

Moderate ambiguity
    ↓
CNN learns useful field-level patterns
    ↓
Meaningful improvement

High ambiguity
    ↓
More expressive models become useful
    ↓
Cross-field / cross-record interactions matter more
```

Compare:

* exact-field baseline
* lightweight learned model
* deeper neural models

Analyze performance as a function of:

* number of exact matching fields
* missing fields
* corrupted fields
* field type
* pair difficulty

The key result should not simply be:

> "Model X achieved the highest F1."

Instead:

> **At what level of record ambiguity does additional model complexity become worthwhile?**

Total cross-entity pairs = 12,495,566

natural ambiguity spectrum:
```text
0 exact: 9,351,748
1 exact: 3,033,620
2 exact:   108,576
3 exact:     1,613
4 exact:         9

                    Cross-entity pairs

0 exact  █████████████████████████████████████████
1 exact  █████████████
2 exact  █
3 exact  .
4 exact  .

                    Same-entity pairs

1 exact  .
2 exact  .
3 exact  █
4 exact  ██
5 exact  ████
6 exact  █████
7 exact  █████
8 exact  ███
9 exact  .
---

# 7. Results

## 7.1 Dataset Characteristics

Report:

* records
* entities
* duplicate rate
* missingness
* exact-match distributions

## 7.2 Blocking Results

Report candidate reduction and recall.

## 7.3 Pairwise Matching

Report model performance.

## 7.4 Graph Resolution

Report cluster-level performance.

## 7.5 Cross-Dataset Comparison

Compare FEBRL-2 and SPIDER v2.

Focus on the relationship between:

```text
Dataset difficulty
        ↓
Exact field agreement
        ↓
Baseline performance
        ↓
Benefit of learned models
        ↓
Benefit of deeper models
```

---

# 8. Discussion

Discuss:

* whether learned representations outperform exact/rule-based matching
* where CNN + MLP provides value
* whether deeper models justify their computational cost
* whether graph refinement improves global consistency
* how dataset characteristics influence model selection

Explicitly discuss cases where a simple method performs sufficiently well.

This is important because the goal is not to argue that deep learning is always necessary.

---

# 9. Limitations

Potential limitations include:

* synthetic benchmark datasets
* limited number of fields
* controlled corruption processes
* dependence on blocking quality
* computational cost of deeper models
* graph methods depending on pairwise prediction quality

---

# 10. Conclusion

Summarize the main findings in terms of the two research questions.

The conclusion should answer:

1. Whether learned pairwise similarity improves noisy multi-record entity resolution.
2. Whether graph refinement improves entity-level consistency.
3. When additional neural model complexity is justified based on dataset difficulty.
