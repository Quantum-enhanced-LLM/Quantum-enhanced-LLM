#Requires -Version 5.1

# --- 1. Argument and Prerequisite Check ---
param (
    [Parameter(Mandatory=$true, HelpMessage="Please provide the Conda environment name.")]
    [string]$CondaEnvName
)

# Stop the script if any command fails. This is the PowerShell equivalent of 'set -e'.
$ErrorActionPreference = 'Stop'

# Check if essential commands 'git' and 'conda' are available.
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Error: 'git' not found. Please install Git and ensure it's in your system's PATH." -ForegroundColor Red
    exit 1
}
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "Error: 'conda' not found. Please install Anaconda/Miniconda and ensure it's in your system's PATH." -ForegroundColor Red
    exit 1
}

# --- 2. Define Repositories to Clone and Install ---
$Repositories = @{
    "QLLM-LLaMA-Factory" = "https://github.com/Quantum-enhanced-LLM/QLLM-LLaMA-Factory.git"
    "QLLM-quanta"        = "https://github.com/Quantum-enhanced-LLM/QLLM-quanta.git"
    "QLLM-peft"          = "https://github.com/Quantum-enhanced-LLM/QLLM-peft.git"
    "QLLM-torchquantum"  = "https://github.com/Quantum-enhanced-LLM/QLLM-torchquantum.git"
}
$TargetBranch = "qllm-develop"

Write-Host ">>> Target Conda environment: '$CondaEnvName'" -ForegroundColor Cyan
Write-Host ">>> Target Git branch for all repos: '$TargetBranch'" -ForegroundColor Cyan

# --- 3. Verify Conda Environment Exists ---
Write-Host ""
Write-Host ">>> Verifying Conda environment..." -ForegroundColor Green
$envList = conda env list | Out-String
if ($envList -notmatch "\b$CondaEnvName\b") {
    Write-Host "Error: Conda environment '$CondaEnvName' does not exist." -ForegroundColor Red
    Write-Host "Please create it first, e.g., 'conda create --name $CondaEnvName python=3.9'" -ForegroundColor Yellow
    exit 1
}
Write-Host ">>> Conda environment '$CondaEnvName' found."

# --- 4. Prerequisite Check: Verify PyTorch with CUDA ---
Write-Host ""
Write-Host ">>> Verifying PyTorch and CUDA availability in '$CondaEnvName'..." -ForegroundColor Green
try {
    conda run -n $CondaEnvName --no-capture-output python -c "import torch; assert torch.cuda.is_available(), 'PyTorch CUDA is not available!'"
    $torchVersion = conda run -n $CondaEnvName python -c "import torch; print(torch.__version__)"
    Write-Host ">>> Verification successful: PyTorch with CUDA support is installed and available." -ForegroundColor Green
    Write-Host ">>> PyTorch version: $torchVersion"
} catch {
    Write-Host "Error: PyTorch with CUDA support not found or not working in '$CondaEnvName'." -ForegroundColor Red
    Write-Host "Please install a CUDA-compatible version of PyTorch before running this script." -ForegroundColor Yellow
    exit 1
}

# --- 5. Install Other Base Dependencies ---
Write-Host ""
Write-Host ">>> Installing other base dependencies (wandb, jieba, etc.)..." -ForegroundColor Green
try {
    conda run -n $CondaEnvName --no-capture-output pip install tf_keras wandb jieba nltk rouge_chinese pyqpanda3 python-dotenv
    Write-Host ">>> Base dependencies installed successfully."
} catch {
    Write-Host "Error: Failed to install base dependencies." -ForegroundColor Red
    exit 1
}

# --- 6. Clone and Install Repositories ---
Write-Host ""
Write-Host ">>> Cloning repositories and installing in editable mode..." -ForegroundColor Green

# **FIX:** Store the starting directory before the loop
$initialDirectory = Get-Location

foreach ($repo in $Repositories.GetEnumerator()) {
    $repoName = $repo.Name
    $repoUrl = $repo.Value

    Write-Host "--------------------------------------------------" -ForegroundColor DarkGray
    Write-Host ">>> Processing repository: '$repoName'" -ForegroundColor Cyan

    try {
        # Clone the repository if it doesn't exist locally
        if (Test-Path -Path $repoName -PathType Container) {
            Write-Host ">>> Directory '$repoName' already exists. Skipping clone." -ForegroundColor Yellow
        } else {
            Write-Host ">>> Cloning '$repoName' from '$repoUrl'..."
            git clone --depth 1 --branch $TargetBranch $repoUrl $repoName
            Write-Host ">>> Clone complete."
        }

        # Change to the repository directory
        Set-Location -Path $repoName

        # Special handling for QLLM-quanta
        if ($repoName -eq "QLLM-quanta") {
            if (Test-Path -Path "quanta" -PathType Container) {
                Write-Host ">>> Entering special subdirectory 'quanta' for installation..."
                Set-Location -Path "quanta"
            } else {
                Write-Host "Error: Expected subdirectory 'quanta' not found in '$repoName'. Cannot install." -ForegroundColor Red
                throw "Installation path not found for QLLM-quanta"
            }
        }

        Write-Host ">>> Installing package in editable mode (pip install -e .)..."
        conda run -n $CondaEnvName --no-capture-output pip install -e .
        Write-Host ">>> Successfully installed '$repoName'." -ForegroundColor Green
    }
    catch {
        Write-Host "Error: An error occurred while processing '$repoName'." -ForegroundColor Red
        # The finally block will still execute to clean up the directory change
        exit 1
    }
    finally {
        # **FIX:** Always return to the initial directory at the end of each loop iteration
        Write-Host ">>> Returning to base directory: '$($initialDirectory.Path)'"
        Set-Location -Path $initialDirectory
    }
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host ">>> All tasks completed successfully!" -ForegroundColor Green
Write-Host ">>> All specified repositories are installed in the '$CondaEnvName' environment." -ForegroundColor Green
Write-Host "=================================================="

exit 0