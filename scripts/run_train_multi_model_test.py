import os
from dotenv import load_dotenv
from pathlib import Path

from scan_utils import scan_parameters, ScanMode

print('load_dotenv', load_dotenv(Path.cwd() / '.env'))

MODEL_BASE_PATH: str = os.getenv('MODEL_BASE_PATH')
DATA_BASE_PATH: str = os.getenv('DATA_ENV_PATH')

if MODEL_BASE_PATH is None or DATA_BASE_PATH is None:
    print('Error: MODEL_BASE_PATH or DATA_ENV_PATH is not set in the .env file.')
    exit(1)


if __name__ == "__main__":
    models = [
        #"ZhipuAI/glm-4-9b-chat-hf",
        #"LLM-Research/Meta-Llama-3-8B-Instruct"
        "Qwen3-0.6B",
        # "Qwen3-1.7B",
        # "Qwen3-4B",
        # "Qwen3-8B",
        # "DeepSeek-R1-0528-Qwen3-8B",
    ]

    for model in models:
        base_arguments = {
            "project_name": "MultiModel",
            "model_name_or_path": MODEL_BASE_PATH + model,
            "dataset_dir": DATA_BASE_PATH,
            "learning_rate": 5.0e-5,
            "num_train_epochs": 3.0,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "max_samples": 3000,
            "lora_target": "q_proj,v_proj",
            "dataset": "CPsyCounD_eval",
            "bf16": True,
            "overwrite_output_dir": False,
            "base_output_path": './qpeft',
            "logging_steps": 1,
            "save_steps": 9999999,
            "eval_steps": 30,
        }

        parameter_scan_grid1 = {
            "lora_rank": [4, 6, 8, 10],
            "finetuning_type_special": ['qpeft'],
            "dataset": [
                "CPsyCounD_eval", 
                "GSM8K_MAIN_TRAIN"
            ],
            "qpeft_arch": [
                'BC', 
                # 'C',  # the quantum-only model
                # 'BD', # NonLinearA
                # 'BE', # NonLinearB
                # 'BF', # NonLinearC
            ],
        }
        
        parameter_scan_grid2 = {
            "lora_rank": [4, 6, 8, 10],
            "finetuning_type_special": ['none'],
            "dataset": [
                "CPsyCounD_eval", 
                "GSM8K_MAIN_TRAIN"
            ],
        }
        
        parameter_scan_grid3 = {
            "lora_rank": [4],
            "finetuning_type_special": ['quanta'],
            "dataset":  [
                "CPsyCounD_eval", 
                "GSM8K_MAIN_TRAIN"
            ],
        }

        parameter_scan_grids = [
            parameter_scan_grid1, 
            parameter_scan_grid2, 
            parameter_scan_grid3
        ]

        # scan_mode = ScanMode.DRY_RUN
        scan_mode = ScanMode.SAVE_CONFIG
        # scan_mode = ScanMode.TRAIN


        for param_scan_grid in parameter_scan_grids:
            scan_parameters(
                param_grid=param_scan_grid,
                base_config_args=base_arguments,
                scan_mode=scan_mode
            )
    