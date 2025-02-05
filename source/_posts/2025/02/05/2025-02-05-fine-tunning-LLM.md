---
title: fine tunning LLM
tags:
  - null
category:
  - null
date: 2025-02-05 17:23:00
---

## Template

```text
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
```

## Fine Tunning

> Using Colab and Ollama

### Install `unsloth`

```python
%%capture
# Using pip install unsloth will take 3 minutes, whilst the below takes <1 minute:
!pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton
!pip install --no-deps cut_cross_entropy unsloth_zoo
!pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
!pip install --no-deps unsloth
```

### Select pretrained model

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",
    max_seq_length=max_seq_length,
    dtype = dtype,
    load_in_4bit= load_in_4bit
)
```

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
```

```python
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Company database: {}

### Input:
SQL Prompt:{}

### Response:
SQL: {}

Explanation: {}
"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    company_databases = examples["sql_context"]
    prompts = examples["sql_prompt"]
    sqls = examples["sql"]
    explanations = examples["sql_explanation"]
    texts = []
    for company_database, prompt, sql, explanation in zip(company_databases, prompts, sqls, explanations):
        text = alpaca_prompt.format(company_database, prompt, sql, explanation) + EOS_TOKEN
        texts.append(text)
    return {"text": texts, }
pass

from datasets import load_dataset
dataset = load_dataset("gretelai/synthetic_text_to_sql", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True)
```

```python
dataset['text'][0]
```

```python
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
trainer_stats = trainer.train()
```

### Upload model to Huggingface

```python
from google.colab import userdata
print(userdata.get('HUGGINGFACE_TOKEN'))

# Install ollama
!curl -fsSL https://ollama.com/install.sh | sh

!git clone https://github.com/ggerganov/llama.cpp.git
!(cd llama.cpp; cmake -B build;cmake --build build --config Release)
!cp llama.cpp/build/bin/llama-quantize llama.cpp/

model.push_to_hub_gguf("desonglll/fine_tuning_text_to_sql_based_on_llama3.1", tokenizer, quantization_method = "f16", token = userdata.get('HUGGINGFACE_TOKEN'))
```

### Run model using ollama

```python
import subprocess

subprocess.Popen(["ollama", "serve"])
import time

time.sleep(3)  # Wait for a few seconds for Ollama to load!

!ollama run hf.co/desonglll/fine_tuning_text_to_sql_based_on_llama3.1
```