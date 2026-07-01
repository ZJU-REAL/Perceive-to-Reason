---
license: apache-2.0
---

  <a href="" target="_blank">
      <img alt="arXiv" src="https://img.shields.io/badge/arXiv-Perceive--to--Reason-red?logo=arxiv" height="20" />
  </a>
  <a href="https://github.com/ZJU-REAL/Perceive-to-Reason" target="_blank">
      <img alt="Code" src="https://img.shields.io/badge/Code-Perceive--to--Reason-white?logo=github" height="20" />
  </a>
  <a href="https://huggingface.co/hongxingli/P2R-4B" target="_blank">
      <img alt="Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-P2R--4B-ffc107?color=ffc107&logoColor=white" height="20" />
  </a>

# P2R-10k

This repository contains the training dataset used in the paper [Perceive-to-Reason: Decoupling Perception and Reasoning for Fine-Grained Visual Reasoning]().

## Dataset Description

P2R-10k is a 10k-sample fine-grained visual reasoning dataset curated for training P2R models with PRA-GRPO. Each sample consists of a high-resolution image and a question that requires fine-grained perception and reasoning to answer.

## Data Sources

P2R-10k is constructed by randomly sampling from the following datasets:

| Source | Samples |
|--------|---------|
| [DeepEyes_train_4K](https://huggingface.co/datasets/Mini-o3/DeepEyes_train_4K) | 3k |
| [VisualProbe_train](https://huggingface.co/datasets/Mini-o3/VisualProbe_train) | 3k |
| [ZwZ-RL-VQA-mini](https://huggingface.co/datasets/muyuho/ZwZ-RL-VQA-mini) | 4k |

## Citation

```bibtex
```
