import os
import subprocess
from pathlib import Path

# --- Slurm Job Script Template ---
SLURM_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=qpeft_eval
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --mem=200G

# Set WANDB_MODE to offline to prevent unnecessary logging or network requests.
export WANDB_MODE=offline

# Execute the evaluation command
echo "Command: {full_command}"
{full_command}

echo "Evaluation finished."
exit 0
"""

from dotenv import load_dotenv
from pathlib import Path

print('load_dotenv status:', load_dotenv(Path.cwd() / '.env'))

MODEL_BASE_PATH: str = os.getenv('MODEL_BASE_PATH')
DATA_BASE_PATH: str = os.getenv('DATA_ENV_PATH')

if MODEL_BASE_PATH is None or DATA_BASE_PATH is None:
    print('Error: MODEL_BASE_PATH or DATA_ENV_PATH is not set in the .env file.')
    exit(1)

job_counter = 0

def generate_slurm_script(
    full_command: str,
    config_info: str # For logging; a description of the model/adapter/settings combination.
) -> str:
    """Generates the content for a single Slurm job script."""
    script_content = SLURM_SCRIPT_TEMPLATE.format(
        full_command=full_command,
        config_info=config_info
    )
    return script_content

def submit_eval_jobs(
    basepath_str: str,
    model_list: list,
    dataset_dir: str,
    train_dataset: str,
    test_dataset: str,
    save_basepath: str = 'eval'
):
    """
    Iterates through adapters in the base path, submitting a Slurm evaluation job
    for each adapter-model combination.
    """
    basepath = Path(basepath_str)
    adapter_path_list = []
    for dirpath, dirnames, filenames in os.walk(basepath):
        for dirname in dirnames:
            full_adapter_path = str(Path(dirpath) / dirname)
            if not dirname.startswith('lora'):
                continue
            if train_dataset in full_adapter_path:
                adapter_path_list.append(full_adapter_path)

    if not adapter_path_list:
        print(f"No adapter paths found in '{basepath_str}' matching train_dataset '{train_dataset}'.")
        return

    print(f"Found {len(adapter_path_list)} adapter paths.")
    adapter_path_list.sort() # Sort to ensure consistent submission order.

    global job_counter

    for model in model_list:
        for adapter_path in adapter_path_list:
            job_counter += 1
            
            # Try to parse the model name from the adapter_path to match it with the model_path.
            # Assuming adapter_path format is: {basepath_str}/{peft_type}/{model_name}/...
            relative_adapter_path_parts = adapter_path.split(os.path.sep)
            
            # Find the position of basepath_str in adapter_path and take the next part as the model_name.
            try:
                base_idx = relative_adapter_path_parts.index(basepath_str.split(os.path.sep)[-1])
                model_name_in_adapter = relative_adapter_path_parts[base_idx + 2]
            except (ValueError, IndexError):
                print(f"Warning: Could not parse model name from adapter path '{adapter_path}'. Skipping.")
                continue

            # Check if the model_path matches the model name in the adapter_path.
            expected_model_dir_name = model.split(os.path.sep)[-1]
            if expected_model_dir_name != model_name_in_adapter:
                # print(f"Skipping: Model name '{expected_model_dir_name}' from model_path does not match name in adapter path '{adapter_path}'.")
                # print('model_name_in_adapter: ', model_name_in_adapter)
                continue # Skip non-matching models.
            
            # Construct the save directory for mmlu results.
            # Example: eval/mmlu/qpeft_BC_1_all/Qwen3-1.7B/CPsyCounD_eval/samples_3000/lora_r8_q_proj_v_proj
            save_dir_relative_to_script = os.path.join(save_basepath, "mmlu", adapter_path[len(basepath_str) + 1:])
            savepath = save_dir_relative_to_script
            full_save_file = os.path.join(savepath, f"all_results.json")
            
            # Check if the result file already exists, and skip if it does.
            if os.path.exists(full_save_file):
                print(f"  Result file '{full_save_file}' already exists. Skipping.")
                continue

            # --- Parse adapter_path to get other parameters ---
            attributes = adapter_path.split(os.path.sep)
            # Get the directory name right before the model_name.
            peft_type = attributes[attributes.index(model_name_in_adapter) - 1] 
            
            try:
                lora_rank_str = next(attr for attr in attributes if attr.startswith('lora_r'))
                lora_rank = int(lora_rank_str.split('_')[1][1:]) # Extract X from rX
            except StopIteration:
                print(f"Warning: Could not extract lora_rank from '{adapter_path}'. Skipping.")
                continue
            except ValueError:
                print(f"Warning: Could not parse lora_rank from '{adapter_path}'. Skipping.")
                continue

            use_quanta = False
            use_qpeft = False            
            qpeft_qcircuit_layers = None
            qpeft_arch = 'ABC'

            # Determine use_quanta and use_qpeft
            if 'qpeft' in peft_type:
                use_qpeft = True
                qpeft_args = peft_type.split('_')
                if len(qpeft_args) > 1:
                    qpeft_arch = qpeft_args[1]
                # Handling of qpeft_qcircuit_layers (if needed)
                if len(qpeft_args) > 2 and qpeft_args[2] != 'default':
                    qpeft_qcircuit_layers = int(qpeft_args[2])                
            elif 'quanta' in peft_type:
                use_quanta = True

            # Create the save directory if it doesn't exist.
            os.makedirs(savepath, exist_ok=True)
            
            # --- Build the evaluation command ---
            # Use llamafactory-cli train --do_predict for evaluation
            command_parts = [
                f"llamafactory-cli train", 
                f"--stage sft", # Evaluation is typically done in the 'sft' stage
                f"--do_predict",
                f"--predict_with_generate", # Ensure generative prediction is enabled
                f"--model_name_or_path {MODEL_BASE_PATH}{model}",
                f"--adapter_name_or_path \"{adapter_path}\"", # adapter_path might contain spaces, so wrap in quotes
                f"--output_dir \"{savepath}\"", # Directory for llamafactory-cli to store results
                f"--eval_dataset {test_dataset}",
                f"--dataset_dir {dataset_dir}",
                f"--lora_target q_proj,v_proj", # Adjust according to your model architecture
                f"--use_quanta {str(use_quanta).lower()}", # Pass "true" or "false"
                f"--use_qpeft {str(use_qpeft).lower()}",   # Pass "true" or "false"
                f"--qpeft_arch {qpeft_arch}",
                f"--qpeft_qcircuit_layers {qpeft_qcircuit_layers}" if qpeft_qcircuit_layers is not None else "",
                f"--lora_rank {lora_rank}",
                f"--max_samples 10", # Max samples for a quick test
                f"--per_device_eval_batch_size 1", # Batch size for single-GPU prediction
            ]
            # Filter out parameters that are None or empty strings
            full_command = " ".join(filter(None, command_parts))

            # Generate Slurm script content using the template
            slurm_script_content = generate_slurm_script(
                full_command=full_command,
                config_info=adapter_path
            )

            # Create a temporary .sh file for submission
            script_filename = f"submit_eval_job_{job_counter}.sh"
            with open(script_filename, "w") as f:
                f.write(slurm_script_content)

            # Submit the script using sbatch
            print(f"Submitting job {job_counter} for {model} and adapter: {adapter_path}")
            
            # continue # Uncomment to dry-run without submitting
            try:
                # Use subprocess to run the sbatch command
                result = subprocess.run(["sbatch", script_filename],
                                        capture_output=True, text=True, check=True)
                job_id = result.stdout.strip()
                print(f"  Submitted job {job_id}")
            except subprocess.CalledProcessError as e:
                print(f"  Error submitting job {script_filename}:")
                print(f"  Return code: {e.returncode}")
                print(f"  Stderr: {e.stderr}")
            except FileNotFoundError:
                print("  Error: 'sbatch' command not found. Is Slurm installed and in your PATH?")


    print(f"\n--- Finished processing. Attempted to submit {job_counter} jobs. ---")


# Adapters' base path
BASE_ADAPTER_PATH_STR = "./qpeft" 

MODEL_LIST = [
    "ZhipuAI/glm-4-9b-chat-hf",
    "LLM-Research/Meta-Llama-3-8B-Instruct",
    "Qwen3-1.7B",
]

DATASET_LIST = [
    ('CPsyCounD_eval', 'CPsyCounD_test_100_0'),
    ('GSM8K_MAIN_TRAIN', 'GSM8K_MAIN_TEST')
]

if __name__ == "__main__":            
    for train_dataset, test_dataset in DATASET_LIST:
        submit_eval_jobs(
            basepath_str=BASE_ADAPTER_PATH_STR,
            model_list=MODEL_LIST,
            dataset_dir=DATA_BASE_PATH,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            save_basepath='eval'
        )