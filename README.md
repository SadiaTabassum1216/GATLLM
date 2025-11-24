# GAT-LLM: A Graph Attention-Based Framework with Large Language Models for Spatio-Temporal Traffic Forecasting

This repository contains the official implementation of **GAT-LLM**, proposed in the paper:

**"GAT-LLM: A Graph Attention-Based Framework with Large Language Models for Spatio-Temporal Traffic Forecasting"**

**Authors:** *Sadia Tabassum, Sumon Ahmed, Naushin Nower*

Paper:
[https://github.com/SadiaTabassum1216/GATLLM/blob/master/GATLLM.pdf](https://github.com/SadiaTabassum1216/GATLLM/blob/master/GATLLM.pdf)

---

## Abstract

Traffic forecasting is a crucial component of intelligent transportation systems (ITS), enabling congestion prevention, route optimization, and effective mobility planning. This work introduces **GAT-LLM**, a hybrid architecture that integrates **Graph Attention Networks (GATs)** for spatial representation with a **partially frozen Large Language Model (LLM)** for temporal sequence modeling.

GATs capture spatial dependencies across traffic locations, while the LLM models long-range temporal patterns. We develop a unified embedding strategy that fuses graph-derived spatial features, temporal encodings, and positional embeddings to form a rich spatio-temporal representation. Preliminary experiments on benchmark datasets show that GAT-LLM outperforms traditional graph-based, attention-based, and LLM-based forecasting models. This work highlights the potential of combining GNNs and LLMs for expressive and transferable spatio-temporal forecasting.

---

## Features

* Graph Attention Network (GAT) for spatial feature extraction
* Pretrained LLM for long-range temporal reasoning
* Unified fusion of spatial, temporal, and positional embeddings
* Transformer-compatible input formatting
* Benchmark evaluation on METR-LA, PEMS, and similar datasets
* Outperforms existing graph-based and LLM-based models

---

## Installation

```bash
git clone https://github.com/SadiaTabassum1216/GATLLM.git
cd GATLLM

conda create -n gatllm python=3.8
conda env list
conda activate gatllm
pip install -r requirements.txt
```

---

## Training

```bash
python train.py --data taxi_drop > taxi_drop_train.log
```

---

## Evaluation

```bash
python test.py --data taxi_drop --checkpoint ./logs/xtaxi_drop/best_model.pth > taxi_drop_test.log
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{tabassum2025gatllm,
  title={GAT-LLM: A Graph Attention-Based Framework with Large Language Models for Spatio-Temporal Traffic Forecasting},
  author={Tabassum, Sadia and Ahmed, Sumon and Nower, Naushin},
  year={2025}
}
```

---
