# Unsupervised Requirements Classification

This repository provides an experimental framework for studying **embedding-based unsupervised classification of software requirements**.

The goal of this project is to investigate how well modern text embeddings capture semantic structure in requirement statements and whether they can be used to perform requirement classification without labeled training data.

The framework supports multiple datasets, embedding models, clustering algorithms, and evaluation pipelines for systematic experimentation.

---

# Motivation

Software requirements classification is typically performed using supervised learning methods that require labeled datasets.

However, labeled requirements data is expensive and difficult to obtain. This project explores whether **unsupervised clustering using modern text embeddings** can approximate supervised classification performance.

The experiments aim to answer the following questions:

- How well do modern sentence embeddings represent requirement semantics?
- How does clustering performance degrade as the number of requirement categories increases?
- Which embedding models perform best for requirement classification?
- How large is the gap between supervised and unsupervised approaches?

---

# Supported Datasets

The framework currently supports multiple requirement datasets.

PROMISE  
Software requirements dataset with multiple requirement categories.

CrowdRE  
Crowdsourced requirements dataset containing short requirement statements.

SecReq  
Security requirements dataset used for initial experiments.

The system is designed to be **dataset-agnostic**, allowing new datasets to be easily integrated.

---

# Embedding Models

The framework evaluates nine embedding models.

Transformer-based embeddings

- SBERT
- SRoBERTa
- MPNet
- E5
- Instructor

Static word embeddings

- Word2Vec (self-trained)
- Word2Vec (GoogleNews pretrained)
- GloVe
- FastText

These embeddings are used to convert requirement statements into vector representations before clustering.

---

# Methods

## Unsupervised Classification

Requirement embeddings are clustered using:

- KMeans
- Hierarchical Agglomerative Clustering (HAC)

For each dataset, the framework evaluates **all combinations of class subsets**. This allows measuring how clustering performance changes as the number of classes increases.

Evaluation metric:

Macro F1 score.

---

## Supervised Baseline

A supervised baseline is implemented using:

TF-IDF + Logistic Regression

This provides an upper bound for comparison against unsupervised clustering.

---

# Experimental Design

For each dataset the system performs:

1. Text preprocessing
2. Embedding generation
3. Clustering
4. Cluster-to-label mapping
5. Macro F1 evaluation
6. Aggregated analysis

Experiments are executed across:

- multiple embeddings
- clustering algorithms
- all class combinations
- multiple datasets

The pipeline also supports **resume-safe experiment execution**, allowing long experiments to continue from previous results.

---

# Analysis Pipeline

The framework includes a full analysis module that produces:

Degradation curves  
Performance vs number of classes

Supervised vs unsupervised comparison  

Embedding ranking  

Robustness metrics  

Binary class difficulty analysis  

Clustering method comparison  

These analyses help understand the strengths and limitations of embedding-based requirement clustering.

---

# Repository Structure

```
data/                datasets used in experiments
datasets/            dataset loaders
embeddings/          embedding implementations
preprocessing/       text cleaning and preprocessing
clustering/          clustering algorithms
evaluation/          evaluation metrics
experiments/         experiment logging
results/             experiment outputs and analysis
```
---
# Running Experiments

Unsupervised experiments

```
python run_full_unsupervised.py --dataset promise --path data/PROMISE_exp.arff
```

Supervised baseline

```
python run_logistic_exhaustive.py --dataset promise --path data/PROMISE_exp.arff
```

Run full analysis

```
python analysis_full.py --dataset promise
```

---

# Notes

Pretrained embedding models are **not included in this repository** due to size constraints.

Results included in the repository are example outputs generated during experiments.

---

# Research Status

This repository is part of an ongoing research project exploring embedding-based methods for software requirements classification.

Further experiments and analysis are currently in progress.
