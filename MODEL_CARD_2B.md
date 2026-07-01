---
license: apache-2.0
base_model:
- Qwen/Qwen3-VL-2B-Instruct
pipeline_tag: image-text-to-text
---

  <a href="" target="_blank">
      <img alt="arXiv" src="https://img.shields.io/badge/arXiv-Perceive--to--Reason-red?logo=arxiv" height="20" />
  </a>
  <a href="https://github.com/ZJU-REAL/Perceive-to-Reason" target="_blank">
      <img alt="Code" src="https://img.shields.io/badge/Code-Perceive--to--Reason-white?logo=github" height="20" />
  </a>
  <a href="https://huggingface.co/datasets/hongxingli/P2R-10k" target="_blank">
      <img alt="Data" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Data-P2R--10k-ffc107?color=ffc107&logoColor=white" height="20" />
  </a>

# P2R-2B

This repository contains the P2R-2B, introduced in [Perceive-to-Reason: Decoupling Perception and Reasoning for Fine-Grained Visual Reasoning]().

## Model Description

P2R-2B is a fine-grained visual reasoning model built upon [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct). It performs inference under the P2R framework, a two-stage visual reasoning framework that decouples perception from reasoning. Training is powered by PRA-GRPO, a role-aware alternating RL strategy.

## Model Performance

| Model | V-Star | HR-Bench-4K | HR-Bench-8K | MME-RealWorld-Lite |
|-------|--------|-------------|-------------|--------------------|
| Qwen3-VL-Instruct-2B | 74.9 | 70.4 | 64.6 | 47.3 |
| **P2R-2B** | **84.3** | **75.1** | **74.8** | **51.3** |
| *Δ* | *+9.4* | *+4.7* | *+10.2* | *+4.0* |

## Usage

```python
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

model = Qwen3VLForConditionalGeneration.from_pretrained("hongxingli/P2R-2B")
processor = AutoProcessor.from_pretrained("hongxingli/P2R-2B")
```

For the full two-stage P2R inference pipeline, please refer to our [code repository](https://github.com/ZJU-REAL/Perceive-to-Reason).

## Citation

```bibtex
```
