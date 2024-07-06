# Phi 3 Fine Tuned using the Orca Math Dataset

## Introduction

Phi-3, a large language model, can be enhanced for mathematical problem-solving by fine-tuning it on the Orca-Math dataset. This dataset, specifically designed for training small language models in math, offers a collection of synthetically generated word problems. By training Phi-3 on these problems, the model learns to tackle mathematical tasks like those found on grade-school math exams. This fine-tuning process improves Phi-3's accuracy in solving word problems without requiring complex techniques like code generation or external tools.

## Model Details

Base Model: [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

Fine Tuned Model: [Ebullioscopic/phi_3_math_fine_tuned](https://huggingface.co/Ebullioscopic/phi_3_math_fine_tuned)

## Dataset Details

Dataset: [Ebullioscopic/orca-math-word-problems-200k](https://huggingface.co/Ebullioscopic/orca-math-word-problems-200k)

## Fine Tuning Details

Trainer type: [SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)

Trainer Module: [Unsloth](https://huggingface.co/unsloth)

Trainer Args: 
```python
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
        max_steps = 1000,
        #num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
```

## Run locally
```bash
ollama run Ebullioscopic/phi_3_math_finetuned
```

## Run on Colab
```

```
## Fine Tuning

Colab Notebook: []()
