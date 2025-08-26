# CURP: Codebook-based Continuous User Representation for Personalized Generation with LLMs

## Introduction

This repository contains the implementation of CURP (Codebook-based Continuous User Representation for Personalized Generation with LLMs), a novel framework for learning interpretable and continuous user representations to enhance personalized text generation with large language models (LLMs).In this work, we propose a bidirectional user encoder and a learnable, semantically initialized discrete codebook to model multi-dimensional user profiles. CURP first aligns the representation space of the bidirectional user encoder with the embedding space of the LLM through Semantic Space Alignment (SSA) on a general corpus, achieving a compression ratio of 4x. Then, the codebook interacts with the user representation via cross-attention to build an interpretable user model. The LLM is further fine-tuned using LoRA to learn how to effectively utilize this user model — a process we refer to as User-Based Codebook Conditioning (UBC). Finally,extensive experiments across diverse personalized generation tasks demonstrate that CURP achieves superior performance and strong generalization compared to state-of-the-art baselines(CSG). The overall framework is illustrated in the figure follows:


## Installation

```bash
git clone https://github.com/RaidonWong/CURP.git curp
cd curp
pip install -r requirements.txt
```



##  Data Preprocessing

### 1. **SSA Dataset**
- **Source**: [DBPedia-Entities-OpenAI-1M](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M)
- **Field Used**: `"text"`
- **Note**: Any large-scale general corpus can be used.

### 2. **UBC Dataset**
- **Source**: [AlignX](https://huggingface.co/datasets/JinaLeejnl/AlignX)
- **Fields Used**: `"prompt"`, `"chosen"`, `"rejected"`, `"Demographic Information"`, `"User-Generated Content"`
- **Note**: We use a filtered subset with length restrictions. Can be replaced with any user history dataset.

### 3. **CSG Datasets**
- **Sources**: 
  - [LaMP Benchmark](https://lamp-benchmark.github.io/)
  - [LongLaMP Benchmark](https://longlamp-benchmark.github.io/)
- **Tasks**: News Headline, Tweet Paraphrase, Review Writing
- **Preprocessing**:
  - **News Headline**: Filter short texts; use LLaMA-3-8B to judge headline-paragraph association.
  - **Tweet Paraphrase**: Remove noisy entries (e.g., `@<ref>`); use LLM for filter.
  - **Review Writing**: Keep consistent-rating reviews, length filter

The details can be found in /curp/CSG/.../deal_dataset.py
The history is random ordered and we do not rank by RAG or other retrieve methods.

## Usage

- First of all, you need to download the LLaMA-3-8B-Instruct model and RoBERTa-large model
  - [LLaMA-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
  - [RoBERTa-large model](https://huggingface.co/FacebookAI/roberta-large)
Then you need to add two speicial token, `<PAD>` and `<BERT>` and resize the LLM tokenizer embedding input. The `<PAD>` is for padding and the `<BERT>` is for indicating the place to insert our user model.

```python
special_tokens = {
    "additional_special_tokens": ["<PAD>", "<BERT>"]
}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))
```


- Secondly, you need to prepare the dataset and accelerate config files. Data Reference: `SSA_data` in /curp/dataset.py and Model code `SSA_model` in /curp/model.py and run
```bash
accelerate launch   --config_file default_config.yaml  /curp/scripts_SSA.py
```

- Thirdly, you need to prepare a codebook which can be semantically or randomly initialized. If semantically, you need to prepare an excel file, each line indicate a description of dimention and run
  
```bash
python -m curp.deal_codebook
```
Then the codebook entries will be encoded by RoBERTa and we only use the `[CLS]` token to represent the whole sentense.

- After that, you are ready to launch the UBC scripts to train the codebook and fine tune the LLM.
  
```bash
accelerate launch   --config_file default_config.yaml  /curp/scripts_SSA.py
```
- Finally, you can evaluate and use the model to generate on various scenarios.'
  
```bash
python -m curp.CSG.Task.task.py
```

The argument is by default. If you want to change, you can change as you need.
































