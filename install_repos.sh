#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. Argument Check ---
if [ -z "$1" ]; then
    echo "Error: Please provide the Conda environment name as the first argument."
    echo "Usage: $0 <conda_env_name>"
    echo "Example: $0 qllm"
    exit 1
fi

CONDA_ENV_NAME="$1"

# --- 2. Define Repositories to Clone and Install ---
# Using an associative array to map local directory names to their Git URLs.
declare -A REPOSITORIES
REPOSITORIES=(
    ["QLLM-LLaMA-Factory"]="https://github.com/Quantum-enhanced-LLM/QLLM-LLaMA-Factory.git"
    ["QLLM-quanta"]="https://github.com/Quantum-enhanced-LLM/QLLM-quanta.git"
    ["QLLM-peft"]="https://github.com/Quantum-enhanced-LLM/QLLM-peft.git"
    ["QLLM-torchquantum"]="https://github.com/Quantum-enhanced-LLM/QLLM-torchquantum.git"
)
TARGET_BRANCH="qllm-develop"

echo ">>> Target Conda environment: '$CONDA_ENV_NAME'"
echo ">>> Target Git branch for all repos: '$TARGET_BRANCH'"

# --- 3. Activate Conda Environment ---
# Ensure the conda command is available in the script.
__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    # Fallback to common installation paths
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then # Common in Docker
        source "/opt/conda/etc/profile.d/conda.sh"
    else
        echo "Error: Could not find Conda initialization script."
        echo "Please run 'conda init bash' and restart your shell, or manually source conda.sh."
        exit 1
    fi
fi
unset __conda_setup

# Attempt to activate the environment
if conda activate "$CONDA_ENV_NAME"; then
    echo ">>> Successfully activated Conda environment: '$CONDA_ENV_NAME'."
else
    echo "Error: Failed to activate Conda environment '$CONDA_ENV_NAME'."
    echo "Please make sure the environment exists. You can create it with 'conda create -n $CONDA_ENV_NAME python=3.x'."
    exit 1
fi

# --- 4. Prerequisite Check: Verify PyTorch with CUDA ---
echo ""
echo ">>> Verifying PyTorch and CUDA availability..."
# Use python's exit code to check for success (0) or failure (non-zero)
if python -c "import torch; exit(0) if torch.cuda.is_available() else exit(1)"; then
    echo ">>> Verification successful: PyTorch with CUDA support is installed and available."
    echo ">>> PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
    echo ">>> CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
else
    echo "Error: PyTorch with CUDA support not found or not working in '$CONDA_ENV_NAME'."
    echo "Please install a CUDA-compatible version of PyTorch before running this script."
    echo "Example: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    exit 1
fi

# --- 5. Install Other Base Dependencies ---
echo ""
echo ">>> Installing other base dependencies (wandb, jieba, etc.)..."
pip install tf_keras wandb jieba nltk rouge_chinese pyqpanda3 python-dotenv
echo ">>> Base dependencies installed."

# --- 6. Clone and Install Repositories ---
echo ""
echo ">>> Cloning repositories and installing in editable mode..."
for repo_name in "${!REPOSITORIES[@]}"; do
    repo_url=${REPOSITORIES[$repo_name]}
    echo "--------------------------------------------------"
    echo ">>> Processing repository: '$repo_name'"

    # Clone the repository if it doesn't exist locally
    if [ -d "$repo_name" ]; then
        echo ">>> Directory '$repo_name' already exists. Skipping clone."
    else
        echo ">>> Cloning '$repo_name' from '$repo_url'..."
        # Use a shallow clone on the target branch to save time and disk space
        git clone --depth 1 --branch "$TARGET_BRANCH" "$repo_url" "$repo_name"
        echo ">>> Clone complete."
    fi

    # Save current directory before entering the repo folder
    current_dir=$(pwd)
    cd "$repo_name"
    
    # Special handling for qllm-quanta, which has its setup.py in a subdirectory
    if [ "$repo_name" == "QLLM-quanta" ]; then
        if [ -d "quanta" ]; then
            echo ">>> Entering special subdirectory 'quanta' for installation..."
            cd "quanta"
        else
             echo "Error: Expected subdirectory 'quanta' not found in '$repo_name'. Cannot install."
             cd "$current_dir" # Go back before exiting
             exit 1
        fi
    fi

    echo ">>> Installing package in editable mode (pip install -e .)..."
    # The `-e` flag makes the installation editable, linking to the source files.
    pip install -e .
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install '$repo_name'."
        cd "$current_dir" # Ensure we go back to the base directory before exiting
        exit 1
    else
        echo ">>> Successfully installed '$repo_name'."
    fi

    # Return to the original directory
    cd "$current_dir"
    echo "--------------------------------------------------"
done

echo ""
echo "=================================================="
echo ">>> All tasks completed successfully!"
echo ">>> All specified repositories are installed in the '$CONDA_ENV_NAME' environment."
echo "=================================================="

exit 0