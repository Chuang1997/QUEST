# QUEST
QUEST: Efficient Extreme Multi-Label Text Classification with Large Language Models on Commodity Hardware

**Paper:** [ACL_anthology](https://aclanthology.org/2024.findings-emnlp.226.pdf)  

## Abstract  
Extreme multi-label text classification (EMTC) involves predicting multiple labels from a vast pool of candidates based on a user's textual query. While traditional BERT-based methods have shown limited success, large language models (LLMs) have brought new possibilities. It is promising to leverage their remarkable comprehension ability to understand textual queries. However, implementing LLMs is non-trivial for two main reasons:  

1. Real-world EMTC datasets can be extremely large, with candidate product pairs reaching up to ten million in real-world scenarios, which poses significant challenges in data ingestion.  
2. The large size of LLMs makes computation and memory demands prohibitive for EMTC applications.  

To this end, we propose QUEST, a Quantized and Efficient Learning with Sampling Technique. QUEST includes:  
- A tailored hash sampling module that reduces the data volume to one-fourth of its original size  
- Compressive fine-tuning of LLMs with only twenty thousand trainable parameters  
- Extensive experiments demonstrating superior performance with fewer resources  

Enables EMTC on commodity hardware like a single NVIDIA RTX 3090 GPU (24GB memory).

## Implementation Overview

This repository provides a basic implementation of QUEST for extreme multi-label text classification. The core functionality is contained in `QUEST.py`, which implements our quantized prompt tuning approach.

### Key Components

- **Model Architecture**: 
  - Default implementation uses **LLaMA-7B** as the base model
  - Users can substitute other foundation models by modifying `args.model` in the QUEST.py file
  
- **Core Features**:
  ```python
  ├── QUEST.py                # Main implementation file
  ├── model.py                # Quantized prompt training
  ├── evaluation.py           # Recall and precision rate
  └── Dataset/                # The dataset to be used
      ├── link.txt     # [Extreme Multi-label Classification Repo](http://manikvarma.org/downloads/XC/XMLRepository.html)
