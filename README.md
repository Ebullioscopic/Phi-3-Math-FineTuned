# Phi 3 Fine Tuned using the Orca Math Dataset

## Introduction

Phi-3, a large language model, can be enhanced for mathematical problem-solving by fine-tuning it on the Orca-Math dataset. This dataset, specifically designed for training small language models in math, offers a collection of synthetically generated word problems. By training Phi-3 on these problems, the model learns to tackle mathematical tasks like those found on grade-school math exams. This fine-tuning process improves Phi-3's accuracy in solving word problems without requiring complex techniques like code generation or external tools.

![](https://github.com/Ebullioscopic/Phi-3-Math-FineTuned/blob/main/images/graphical.jpg)

![](https://github.com/Ebullioscopic/Phi-3-Math-FineTuned/blob/main/images/ss_1.jpeg)

![](https://github.com/Ebullioscopic/Phi-3-Math-FineTuned/blob/main/images/ss_2.jpeg)

![Command Line](https://github.com/Ebullioscopic/Phi-3-Math-FineTuned/blob/main/images/command_line.png)

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
    packing = False, 
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

Via Ollama:

```bash
ollama run Ebullioscopic/phi_3_math_finetuned
```

## Run on Colab

JuPyter Notebook: [Phi-3-Inference](https://github.com/Ebullioscopic/Phi-3-Math-FineTuned/blob/main/Phi_3_Inference.ipynb)

## Fine Tuning

JuPyter Notebook: [Phi-3-Math-FineTuning](https://github.com/Ebullioscopic/Phi-3-Math-FineTuned/blob/main/Phi_3_Fine_Tuning.ipynb)

## Retrieval Augmented Generation

Fine-tuning large language models like Phi-3 on massive datasets can be a slow crawl. Here's where Retrieval-Augmented Generation (RAG) comes in. RAG acts like a shortcut. It retrieves relevant information (think solved problems!) from a smaller, pre-built knowledge base when presented with a new math problem. Phi-3 then uses this retrieved context to understand and solve the problem itself.  This way, RAG leverages existing knowledge to speed up learning and potentially improve Phi-3's problem-solving accuracy without needing to train on the entire dataset. 

## Creating Virtual Environment

### For Windows
```bat
cd path\to\app
python -m pip install virtualenv
python -m virtualenv venv
venv\Scripts\activate
```
**Alternative: Batch File**

*Save with .bat extension and run*
```bat
@echo off

rem 1. Navigate to the desired directory
cd path\to\app

rem 2. Check if Python is installed
if "%PYTHON%"=="" (
  echo Error: Python is not installed. Please install Python 3 from https://www.python.org/downloads/windows/
  exit /b 1
)

rem 3. Check if virtualenv is installed (using pip)
%PYTHON%\Scripts\pip --version > NUL 2>&1 (
  echo Error: virtualenv is not installed. Installing using pip...
  %PYTHON%\Scripts\pip install virtualenv
)

rem 4. Create the virtual environment named "venv"
%PYTHON%\Scripts\virtualenv venv

rem 5. Activate the virtual environment
venv\Scripts\activate

rem Verification (optional):
where python  # Should show the path to the virtual environment's Python interpreter

rem 6. Install project dependencies (using pip within the activated virtual environment)
pip install -r requirements.txt  # Replace with your specific requirements file if needed

```
### For Linux
```shell
# 1. Navigate to the desired directory where you want to create the virtual environment
cd path/to/app

# 2. Install virtualenv if it's not already present (assuming Python 3 is installed)
python3 -m ensurepip --upgrade  # Installs or upgrades pip if necessary

# 3. Create the virtual environment named "venv"
python3 -m venv venv

# 4. Activate the virtual environment
source venv/bin/activate  # For bash or similar shells

# Verification (optional):
which python  # Should show the path to the virtual environment's Python interpreter

# 5. Install project dependencies (using pip within the activated virtual environment)
pip install -r requirements.txt  # Replace with your specific requirements file if needed

```
### For macOS
```bash
# 1. Navigate to the desired directory where you want to create the virtual environment
cd path/to/app

# 2. Check if pip is installed (Python 3 on macOS usually includes pip)
python3 -m pip --version  # Output indicates pip's presence

# 3. If pip is not installed, use `homebrew` (package manager)
if [ $? -ne 0 ]; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"  # Install Homebrew (if not present)
  brew install python  # Install Python 3 (if not present)
fi

# 4. Create the virtual environment named "venv"
python3 -m venv venv

# 5. Activate the virtual environment
source venv/bin/activate  # For bash or similar shells

# Verification (optional):
which python  # Should show the path to the virtual environment's Python interpreter

# 6. Install project dependencies (using pip within the activated virtual environment)
pip install -r requirements.txt  # Replace with your specific requirements file if needed
```
## Installing Dependencies

### For Windows
```bat
python -m pip install -r requirements.txt
```
### For Linux
```shell
pip3 install -r requirements.txt
```
### For macOS
```bash
pip3 install -r requirements.txt
```
## Configuring the access token

### For Windows
```bat
notepad .env
```
Inside the `.env` file, add a line in the following format
```
HF_TOKEN = your_hugging_face_token
```
Replace `your_hugging_face_token` with your actual Hugging Face access token. You can obtain your token by creating an account and following the instructions here: [User Access Tokens](https://huggingface.co/docs/hub/en/security-tokens)

### Run Locally
Using Streamlit:
```bash
cd path/to/project
python -m pip install -r requirements.txt
streamlit run app/main.py
```
## Architecture Diagram
![Architecture Diagram](https://github.com/Ebullioscopic/Phi-3-Math-FineTuned/blob/main/images/architecture-diagram.jpeg)
## Requirements

- [Unsloth](https://huggingface.co/unsloth)
- [Hugging Face Hub](https://huggingface.co/hub)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/main)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/en/main)
- [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/en/main)
- [Hugging Face Trainer](https://huggingface.co/docs/transformers/en/main/trainers)
- [Hugging Face PyTorch Lightning](https://huggingface.co/docs/transformers/en/main/training/pytorch-lightning)
- [Streamlit](https://streamlit.io/)
- [Ollama]()

## References

- [Jupyter notebooks with code examples for pre-training](https://github.com/aaaastark/Pretrain_Finetune_Transformers_Pytorch)
- [Code and implementation details for BERT](https://github.com/google-research/bert)
- [data loading, tokenization, and training](https://kedion.medium.com/fine-tuning-nlp-models-with-hugging-face-f92d55949b66)
- [Pre-trained Transformers for Text Classification](https://towardsdatascience.com/an-introduction-to-fine-tuning-pre-trained-transformers-models-9ea546611664)
- [BERT for Sentiment Analysis with TensorFlow 2.x and Hugging Face](https://pen_spark.medium.com/comprehensive-guide-to-bert-1e5e33b2b8c8)
- [BERT with PyTorch](https://www.analyticsvidhya.com/blog/2022/11/comprehensive-guide-to-bert/)
- [Text Classification Model](https://github.com/huggingface/notebooks/blob/main/examples/text_classification.ipynb)
- [Question Answering Model with Hugging Face Transformers](https://huggingface.co/transformers/v3.3.1/custom_datasets.html)
- [Transformers and fine-tuning concepts](https://www.coursera.org/specializations/deep-learning)
- [Transformer architecture](https://arxiv.org/abs/1706.03762)
- [BERT for various NLP tasks like question answering and textual entailment, comparing](https://arxiv.org/abs/2012.02462)
- [Neural language models, including transformers](https://arxiv.org/abs/1510.00726)
- [Hierarchical Structures in Transformers](https://arxiv.org/pdf/2210.09221)

## Contributors

[Hariharan Mudaliar](https://www.linkedin.com/in/hariharan-mudaliar)

[Tumati Omkar Chowdary](https://www.linkedin.com/in/omkar-chowdary-84023226a)

[Akula Satya Sesha Sai](https://www.linkedin.com/in/akula-satya-sesha-sai-4994a924b)

## Documentations

[PDF Link](https://github.com/Ebullioscopic/Phi-3-Math-FineTuned/blob/main/INTEL_Orca_Math.pdf)

[PPT Link](https://github.com/Ebullioscopic/Phi-3-Math-FineTuned/blob/main/INTEL_Orca_Math.pptx)
