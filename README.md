# Unsupervised Requirements Classification

This repository contains a dataset-agnostic experimental framework for embedding-based unsupervised and supervised classification of software requirements.

## Features

- Multiple datasets (PROMISE, CrowdRE, SecReq)
- 9 embedding models (transformer + static embeddings)
- KMeans and HAC clustering
- Supervised logistic regression baseline
- Resume-safe experiment execution
- Full evaluation and analysis pipeline

## Run

Unsupervised:
python run_full_unsupervised.py --dataset promise --path data/PROMISE_exp.arff

Supervised:
python run_logistic_exhaustive.py --dataset promise --path data/PROMISE_exp.arff

Analysis:
python analysis_full.py --dataset promise

## Notes

Pretrained embedding models are not included in this repository.
Results included are demonstration outputs for SecReq dataset.