from enum import Enum
import itertools
from typing import Any, Dict, List

from training_config import TrainingConfig, run_training, save_config_file


class ScanMode(Enum):
    DRY_RUN = "dry_run"         # Only print the configuration, do not execute.
    TRAIN = "train"             # Execute the training.
    SAVE_CONFIG = "save_config" # Only save the YAML configuration file.

    
def scan_parameters(
    param_grid: Dict[str, List[Any]],
    base_config_args: Dict[str, Any],
    scan_mode: ScanMode = ScanMode.TRAIN,
):
    """
    Iterates over a parameter grid, creates a configuration for each combination,
    and performs the corresponding action.
    """
    keys = list(param_grid.keys())
    value_combinations = list(itertools.product(*(param_grid[k] for k in keys)))
    
    print(f"Initiating parameter scan with a total of {len(value_combinations)} combinations. Mode: {scan_mode.name}")

    for i, combo_values in enumerate(value_combinations):
        # Create a dictionary for the current parameter combination.
        current_params_override = dict(zip(keys, combo_values))
        
        # Merge base arguments with the current combination to create a config object.
        # Note: **current_params_override will overwrite any matching keys in **base_config_args.
        current_params = {**base_config_args, **current_params_override}
        config = TrainingConfig(**current_params) 
        
        print(f"\n--- [{i+1}/{len(value_combinations)}] ---")
        print(f"Output directory: {config.output_dir}")
        
        if scan_mode == ScanMode.SAVE_CONFIG:
            save_config_file(config)
        elif scan_mode == ScanMode.DRY_RUN:
            # In Dry Run mode, print the full dictionary for preview.
            print("Dry run mode. Configuration details:", config.to_llamafactory_dict())
            continue
        elif scan_mode == ScanMode.TRAIN:
            if not run_training(config):
                print(f"Run failed, aborting scan at: {config.output_dir}")
                # To continue scanning even after a failure, remove the 'break' statement below.
                # break 
        else:
            raise ValueError(f"Invalid scan mode: {scan_mode}")
    
    print("\nParameter scan completed.")