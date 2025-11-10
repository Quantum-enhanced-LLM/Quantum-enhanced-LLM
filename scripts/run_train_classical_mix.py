import os
from scan_utils import scan_parameters, ScanMode
from training_config import MAX_LAYER_DICT

from dotenv import load_dotenv
from pathlib import Path
print('load_dotenv', load_dotenv(Path.cwd() / '.env'))

MODEL_BASE_PATH: str = os.getenv('MODEL_BASE_PATH')
DATA_BASE_PATH: str = os.getenv('DATA_ENV_PATH')

if MODEL_BASE_PATH is None or DATA_BASE_PATH is None:
    print('Error: MODEL_BASE_PATH or DATA_ENV_PATH is not set in the .env file.')
    exit(1)

if __name__ == "__main__":

    models = [
        "Qwen3-0.6B", 
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
    ]
    for model in models:
        max_layer = MAX_LAYER_DICT[model]
        step = 1 if max_layer < 28 else 2

        base_arguments = {
            "project_name": "ClassicalMix",
            "model_name_or_path": MODEL_BASE_PATH + model,
            "dataset_dir": DATA_BASE_PATH,
            "learning_rate": 5.0e-5,
            "dataset": "CPsyCounD_eval", # Fixed dataset for this scan example
            "finetuning_type": "lora",
            "finetuning_type_special": 'qpeft',
            "qpeft_arch": 'BC',
            "lora_target": "q_proj,v_proj",
            "num_train_epochs": 3.0, # Shorter epochs for scan illustration
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "bf16": True,
            "max_samples": 300,
            "overwrite_output_dir": False, # Be careful with this for real scans
            "base_output_path": "./qpeft", # Dedicated base for scans
            "logging_steps": 1,
            "save_steps": 9999999, # Might not want to save checkpoints for every scan run or save less often
            "eval_steps": 9999999,
        }
        
        parameter_scan_grid1 = {
            "lora_rank": [
                4, 
                6, 
                8, 
                10, 
            ],
            "qpeft_classical_mix_method": [
                "head",
                "tail",
            ],
            "qpeft_classical_layers_range": [
                i for i in range(0, max_layer, step)
            ]
        }
        
        parameter_scan_grid2 = {
            "lora_rank": [
                4, 
                6, 
                8, 
                10, 
            ],
            "qpeft_classical_mix_method": [
                "head",
            ],
            "qpeft_classical_layers_range": [
                None
            ]
        }

        parameter_scan_grids = [
            parameter_scan_grid1, 
            parameter_scan_grid2, 
        ]

        # scan_mode = ScanMode.DRY_RUN
        scan_mode = ScanMode.SAVE_CONFIG
        # scan_mode = ScanMode.TRAIN


        # 执行扫描
        for param_scan_grid in parameter_scan_grids:
            scan_parameters(
                param_grid=param_scan_grid,
                base_config_args=base_arguments,
                scan_mode=scan_mode
            )