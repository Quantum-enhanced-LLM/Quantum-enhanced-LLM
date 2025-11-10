# Quantum-enhanced LLM: Synergizing Quantum Machine Learning and Large Language Models

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains the official implementation for the paper **"Quantum-enhanced LLM: Synergizing Quantum Machine Learning and Large Language Models"**. Our work explores the integration of quantum machine learning techniques with large language models to enhance their capabilities.

## Table of Contents
- [Quantum-enhanced LLM: Synergizing Quantum Machine Learning and Large Language Models](#quantum-enhanced-llm-synergizing-quantum-machine-learning-and-large-language-models)
  - [Table of Contents](#table-of-contents)
  - [Quick Start: Tiny Demo](#quick-start-tiny-demo)
    - [System Requirements](#system-requirements)
    - [1. Environment Setup](#1-environment-setup)
    - [2. Download Models and Datasets](#2-download-models-and-datasets)
    - [3. Clone and Install Dependencies](#3-clone-and-install-dependencies)
    - [4. Configure and Run Training](#4-configure-and-run-training)
    - [5. Expected Output](#5-expected-output)
  - [Full-experiment Reproduction](#full-experiment-reproduction)
    - [Prerequisites](#prerequisites)
    - [Base Models](#base-models)
    - [Datasets](#datasets)
  - [Acknowledgements](#acknowledgements)
  - [Citation](#citation)
  - [License](#license)

## Quick Start: Tiny Demo

This guide walks you through a minimal working example to demonstrate the core functionality of our project. The demo involves fine-tuning and evaluating the `Qwen/Qwen3-0.6B` model on the `CPsyCounD` dataset.

### System Requirements
- **OS**: Linux (Recommended) or Windows
- **Python**: 3.10
- **NVIDIA GPU**: CUDA 12.1 or newer is recommended.
- **PyTorch**: Version 2.3.0 or newer with CUDA support.
- **Git & Git LFS**: For cloning models and datasets.

### 1. Environment Setup
We strongly recommend using a virtual environment (e.g., Conda) to manage dependencies.

```bash
# Create and activate a new Conda environment
conda create -n qllm python=3.10
conda activate qllm
```

Next, install PyTorch. Please visit the [PyTorch official website](https://pytorch.org/get-started/locally/) to find the correct command for your specific CUDA version. For example:

```bash
# Example for CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify that PyTorch can access your GPU:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
# Expected output for CUDA available should be True
```

### 2. Download Models and Datasets

First, ensure Git LFS is installed.
```bash
git lfs install
```

Then, clone the base model and dataset from Hugging Face:
```bash
# Download the base model
git clone https://huggingface.co/Qwen/Qwen3-0.6B

# Download the dataset for the demo
git clone https://huggingface.co/datasets/CAS-SIAT-XinHai/CPsyCoun
```

### 3. Clone and Install Dependencies

Clone this repository and install the required packages using the provided scripts.

```bash
git clone https://github.com/Quantum-enhanced-LLM/Quantum-enhanced-LLM
cd Quantum-enhanced-LLM
```

**For Linux:**
```bash
bash install_repos.sh qllm
```

**For Windows (PowerShell):**
```bash
.\install_repos.ps1 -CondaEnvName qllm
```

### 4. Configure and Run Training

It's good practice to create a separate working directory to keep generated files organized.

```bash
# Create and enter a working directory
mkdir ../QLLM-model-test
cd ../QLLM-model-test

# Copy necessary scripts from the repository
cp ../Quantum-enhanced-LLM/scripts/run_train_multi_model_test.py .
cp ../Quantum-enhanced-LLM/scripts/scan_utils.py .
cp ../Quantum-enhanced-LLM/scripts/training_config.py .
```

Create a `.env` file in the `QLLM-model-test` directory with the following content. **Remember to update the paths to match your system.**

```dotenv
# Paths to your downloaded models and datasets.
# IMPORTANT: Paths must end with a forward slash '/'.
MODEL_BASE_PATH=/path/to/your/models/
DATA_ENV_PATH=/path/to/your/data/

# --- Quantum Backend Configuration ---
# Select the quantum backend: torchquantum (default simulator), vqnet, vqnet_virtual, vqnet_noisy
# BACKEND=torchquantum

# Number of measurement shots for quantum circuits.
# When using vqnet/vqnet_virtual, SHOTS must be set.
# SHOTS=0 with BACKEND=vqnet implies ideal simulation (returns probabilities directly).
SHOTS=0

# --- Real Quantum Chip Configuration (if using BACKEND=vqnet) ---
# API_KEY=your_api_key_here
# REAL_CHIP_NAME=WK_C102_400
# CIRCUIT_BATCH_COUNT=20 # Number of circuits submitted as a batch (default: 200)

# --- Debugging and Logging ---
# Uncomment to enable additional logging during training and inference.
# QPEFT_LOG_CALLBACK=1
# OUTPUT_GRAD_DETAILS=1
# OUTPUT_FORWARD_DETAILS=1
# QPEFT_LOG_FORWARD_CALLBACK=1
```

Now, run the script to generate training configuration files.
```bash
python run_train_multi_model_test.py
```

This will create a `configs/` directory. Arrange your downloaded models and datasets so your directory structure looks like this:
```
QLLM-model-test/
├── models/
│   └── Qwen3-0.6B/
├── data/
│   ├── dataset_info.json
│   └── CPsyCounD.json
├── .env
├── configs/
│   └── lora_Qwen3-0.6B_CPsyCounD_eval_samples_3000_lora_r4_q_proj_v_proj.yaml
│   └── ...(other yaml files)
├── run_train_multi_model_test.py
├── scan_utils.py
└── training_config.py
```

Finally, launch the training.

**For Linux:**
```bash
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

llamafactory-cli train configs/lora_Qwen3-0.6B_CPsyCounD_eval_samples_3000_lora_r4_q_proj_v_proj.yaml
```

**For Windows (Command Prompt):**
```bash
set WANDB_MODE=offline
set CUDA_VISIBLE_DEVICES=0

llamafactory-cli train "configs/lora_Qwen3-0.6B_CPsyCounD_eval_samples_3000_lora_r4_q_proj_v_proj.yaml"
```

The training process of LoRA should take approximately 20 minutes on a modern GPU.

### 5. Expected Output
```text
... (training logs)

***** train metrics *****
  epoch                    =        3.0
  total_flos               = 12161262GF
  train_loss               =     2.1591
  train_runtime            = 0:15:56.57
  train_samples_per_second =      8.468
  train_steps_per_second   =      0.267
[INFO|trainer.py:4327] >>> Running Evaluation
[INFO|trainer.py:4329] >>>   Num examples = 300
[INFO|trainer.py:4332] >>>   Batch size = 1
100%|████████████████████████████████████████████████████████████████████| 300/300 [00:16<00:00, 18.16it/s]
***** eval metrics *****
  epoch                   =        3.0
  eval_loss               =     1.9022
  eval_runtime            = 0:00:16.57
  eval_samples_per_second =     18.095
  eval_steps_per_second   =     18.095
[INFO|modelcard.py:450] >>> Dropping the following result as it does not have all the necessary fields:
{'task': {'name': 'Causal Language Modeling', 'type': 'text-generation'}}
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync <local_offline_run_directory>
wandb: Find logs at: <local_offline_run_directory>/logs
```


## Full-experiment Reproduction

To reproduce all experiments from the paper, you will need to download all specified models and datasets.

> **Note:** Downloading all artifacts will require significant disk space and time.

### Prerequisites
Ensure Git LFS is installed to handle large files.
```bash
git lfs install
```
Follow the same [environment setup](#1-environment-setup) as in the quick start guide.

### Base Models

Clone the following model repositories from Hugging Face:

```bash
# Qwen3 Series
git clone https://huggingface.co/Qwen/Qwen3-0.6B
git clone https://huggingface.co/Qwen/Qwen3-1.7B
git clone https://huggingface.co/Qwen/Qwen3-4B
git clone https://huggingface.co/Qwen/Qwen3-8B

# DeepSeek-R1-Qwen3-8B
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

# LLaMA-3.2
git clone https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

# GLM-4
git clone https://huggingface.co/zai-org/glm-4-9b-chat
```

### Datasets

Clone the following dataset repositories from Hugging Face:

```bash
# GSM8K
git clone https://huggingface.co/datasets/openai/gsm8k

# CPsyCoun
git clone https://huggingface.co/datasets/CAS-SIAT-XinHai/CPsyCoun

# Finance-Instruct-500K
git clone https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k

# medical-o1-reasoning-SFT
git clone https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT

# CH-R1-Math
git clone https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k
```
After downloading, update the paths in your `.env` file and use the scripts to generate the corresponding training configurations.

## Acknowledgements

This project is built upon several outstanding open-source libraries. We extend our sincere gratitude to their developers and contributors.
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (Apache License 2.0)
- [PEFT](https://github.com/huggingface/peft) (Apache License 2.0)
- [TorchQuantum](https://github.com/mit-han-lab/torchquantum) (MIT License)
- [Quanta](https://github.com/aitechnology-forward/quanta) (Apache License 2.0)

## Citation

Todo

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for more details.