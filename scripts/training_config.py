from dataclasses import dataclass, field
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Union, Optional

import yaml

# Maximum layer indices for various models
MAX_LAYER_DICT = {
    'Qwen3-0.6B': 27, 
    'Qwen3-1.7B': 27, # from 0-27
    'Qwen3-4B': 35, 
    'Qwen3-8B': 35, # from 0-35
    'Meta-Llama-3-8B-Instruct': 31, # from 0-31
}

# Mapping for QPEFT classical mix types to their character codes
QPEFT_CLASSICAL_MIX_TYPE_DICT = {
    "head": "H", 
    "tail": "T", 
    "random": "R"
}

@dataclass
class TrainingConfig:
    """
    A dataclass for storing and managing all training configurations.
    It handles all parameters and derives values such as output directories
    and filenames based on them.
    """
    # --- Base Model and Data Config ---
    model_name_or_path: str
    dataset: Union[str, List[str]]
    # template is decided automatically based on model_short_name
    # template: str = "qwen" 
    max_samples: int = 1000
    cutoff_len: int = 1024
    
    # --- Fine-tuning Method Config ---
    stage: str = "sft"
    finetuning_type: str = "lora"
    finetuning_type_special: str = 'none' # 'none', 'quanta', or 'qpeft'
    
    # --- LoRA Specific Config ---
    lora_rank: Optional[int] = 4
    lora_target: str = "q_proj,v_proj"
    
    # --- QPEFT Specific Config ---
    qpeft_arch: str = 'BC' # 'C', 'BC', 'BD', 'BE', 'BF'
    qpeft_qcircuit_layers: Optional[Union[int, str]] = None
    qpeft_classical_mix_method: str = "head" # 'head', 'tail', 'random'
    qpeft_classical_layers_range: Optional[int] = None
    #qpeft_log_callback: bool = False

    # --- Path and Naming Config ---
    dataset_dir: Optional[str] = None
    base_output_path: str = "./qpeft"
    output_dir_suffix: str = ""
    
    # --- Training Process Config ---
    learning_rate: float = 5.0e-4
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    logging_steps: int = 1
    save_steps: int = 100
    eval_steps: int = 30
    val_size: float = 0.1
    bf16: bool = True
    
    # --- Advanced and System Config ---
    plot_loss: bool = False
    overwrite_output_dir: bool = False
    overwrite_cache: bool = True
    report_to: str = "wandb"
    per_device_eval_batch_size: int = 1
    eval_strategy: str = "steps"
    trust_remote_code: bool = True
    preprocessing_num_workers: int = 16
    ddp_timeout: int = 180000000
    resume_from_checkpoint: Optional[str] = None
    
    # --- Allow any extra arguments supported by llamafactory ---
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    project_name: Optional[str] = None

    @property
    def model_short_name(self) -> str:
        """Gets the base name of the model, used for constructing paths."""
        return os.path.basename(self.model_name_or_path).replace("/", "_")
    
    @property
    def template(self) -> str:
        """Gets the template name for the model."""
        if self.model_short_name.startswith("Qwen3-"):
            # either reasoning or non-reasoning is decided by dataset name
            REASONING_DATASET = [] # Define your reasoning datasets here
            if self.primary_dataset_name in REASONING_DATASET:
                return "qwen3"
            else:
                return "qwen"
        elif self.model_short_name.startswith("deepseek-"):
            return "deepseek3"
        elif self.model_short_name.startswith("llama3-") or self.model_short_name.startswith("Meta-Llama-3-"):
            return "llama3"
        elif self.model_short_name.startswith("Qwen2.5-"):
            return "qwen"
        elif self.model_short_name.startswith("glm-4"):
            return "glm4"
        else:
            raise ValueError(f"Cannot determine template name: unknown model name '{self.model_short_name}'")

    @property
    def real_base_output_path(self) -> str:
        """Gets the full base output path, including the project name."""
        if not self.project_name:
            raise ValueError("project_name is not set.")
        
        return os.path.join(self.base_output_path, self.project_name)

    @property
    def primary_dataset_name(self) -> str:
        """Gets the primary dataset name, used for constructing paths."""
        if isinstance(self.dataset, list):
            raise ValueError("Multiple datasets are not supported at the moment.")
        if ',' in self.dataset:
            raise ValueError("Multiple datasets specified as a string are not supported.")
        
        return self.dataset
    
    @property
    def lora_target_in_path(self) -> str:
        """Generates a path-safe string for LoRA targets."""
        return self.lora_target.replace(',', '_')

    @property
    def qpeft_config_str(self) -> str:
        """Generates the configuration string for the QPEFT part."""
        if self.finetuning_type_special != 'qpeft':
            return ""
        
        arch = self.qpeft_arch
        n_qlayers = f"{self.qpeft_qcircuit_layers}" if self.qpeft_qcircuit_layers is not None else "default"
        
        if self.qpeft_classical_layers_range is None:
            return f"{arch}_{n_qlayers}_all"
        else:
            method_char = QPEFT_CLASSICAL_MIX_TYPE_DICT.get(self.qpeft_classical_mix_method)
            if not method_char:
                raise ValueError(f"Invalid qpeft_classical_mix_method: {self.qpeft_classical_mix_method}")
            return f"{arch}_{n_qlayers}_{method_char}{self.qpeft_classical_layers_range}"
    
    @property
    def _path_parts(self) -> List[str]:
        """Generates a list of path parts for building directory and file names (private helper method)."""
        lora_type_map = {
            'none': "lora",
            'quanta': "quanta",
            'qpeft': f"qpeft_{self.qpeft_config_str}"
        }
        lora_type = lora_type_map.get(self.finetuning_type_special)
        if not lora_type:
            raise ValueError(f"Invalid finetuning_type_special: {self.finetuning_type_special}")

        parts = [
            lora_type,
            self.model_short_name,
            self.primary_dataset_name,
            f"samples_{self.max_samples}",
        ]
        
        type_str = self.finetuning_type
        if self.finetuning_type == "lora" and self.lora_rank is not None:
            type_str += f"_r{self.lora_rank}"
        
        if self.lora_target is not None:
            # Convert targets to a path-safe string
            lora_target_str = self.lora_target_in_path
            type_str += f"_{lora_target_str}"
        
        parts.append(type_str + self.output_dir_suffix)
        return parts
    
    @property
    def output_dir(self) -> str:
        """Automatically generates a unique output directory path based on the configuration."""
        return os.path.join(self.real_base_output_path, *self._path_parts)
    
    @property
    def config_filename(self) -> str:
        """Automatically generates a unique YAML filename based on the configuration."""
        return "_".join(self._path_parts) + ".yaml"
        
    def generate_classical_mix_layers(self) -> List[int]:
        # Find the corresponding model size from the pre-defined mapping.
        model_size = next((size for size in MAX_LAYER_DICT if self.model_name_or_path.endswith(size)), None)
        if not model_size:
            raise ValueError(f"Cannot determine model size for: {self.model_name_or_path}")
        max_layers = MAX_LAYER_DICT[model_size]

        if self.qpeft_classical_mix_method == 'head':
            layers = list(range(self.qpeft_classical_layers_range + 1))
        elif self.qpeft_classical_mix_method == 'tail':
            start_layer = max_layers - self.qpeft_classical_layers_range
            if start_layer < 0:
                raise ValueError(f"qpeft_classical_layers_range ({self.qpeft_classical_layers_range}) is too large for model {model_size}")
            layers = list(range(start_layer, max_layers + 1))
        else: 
            # 'random' or other future methods can be added here.
            raise ValueError(f"Unsupported qpeft_classical_mix_method: {self.qpeft_classical_mix_method}")
        
        print(f"Generated classical layers for method '{self.qpeft_classical_mix_method}': {layers}")
        return layers


    def to_llamafactory_dict(self) -> Dict[str, Any]:
        """Converts the configuration object into the dictionary format required by llamafactory-cli."""

        # project_name must be set
        if not self.project_name:
            raise ValueError("project_name must be set.")

        # Convert dataclass fields to a dictionary
        config_dict = {
            "model_name_or_path": self.model_name_or_path,
            "dataset_dir": self.dataset_dir,
            "trust_remote_code": self.trust_remote_code,
            "stage": self.stage,
            "do_train": True,
            "finetuning_type": self.finetuning_type,
            "save_safetensors": False,
            "use_quanta": self.finetuning_type_special == 'quanta',
            "use_qpeft": self.finetuning_type_special == 'qpeft',
            "dataset": self.dataset if isinstance(self.dataset, str) else ",".join(self.dataset),
            "template": self.template,
            "cutoff_len": self.cutoff_len,
            "max_samples": self.max_samples,
            "overwrite_cache": self.overwrite_cache,
            "preprocessing_num_workers": self.preprocessing_num_workers,
            "output_dir": self.output_dir, # Use the derived property
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "plot_loss": self.plot_loss,
            "overwrite_output_dir": self.overwrite_output_dir,
            "report_to": self.report_to,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "bf16": self.bf16,
            "ddp_timeout": self.ddp_timeout,
            "val_size": self.val_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
        }

        if self.finetuning_type == "lora":
            config_dict["lora_rank"] = self.lora_rank
            config_dict["lora_target"] = self.lora_target
        else:
            raise ValueError(f"Unsupported finetuning_type: {self.finetuning_type}")
        
        if self.finetuning_type_special == 'qpeft':
            config_dict['qpeft_arch'] = self.qpeft_arch
            if self.qpeft_qcircuit_layers is not None:
                config_dict['qpeft_qcircuit_layers'] = self.qpeft_qcircuit_layers
            if self.qpeft_classical_layers_range is not None:
                # generate classical layers (from range(int) to list of strings)
                layers = self.generate_classical_mix_layers()
                config_dict['qpeft_classical_layers'] = ','.join(map(str, layers))

        if self.resume_from_checkpoint:
            config_dict["resume_from_checkpoint"] = self.resume_from_checkpoint
        
        # Merge any extra kwargs
        config_dict.update(self.extra_kwargs)
        
        return config_dict
    


def run_training(config: TrainingConfig) -> bool:
    """Executes the training process based on a given TrainingConfig object."""
    params_dict = config.to_llamafactory_dict()
    temp_yaml_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False, encoding='utf-8') as tmp_file:
            yaml.dump(params_dict, tmp_file, sort_keys=False, allow_unicode=True)
            temp_yaml_path = tmp_file.name
        
        print(f"--- Starting training ---\nConfig: {config.output_dir}\nYAML: {temp_yaml_path}")
        command = ["llamafactory-cli", "train", temp_yaml_path]
        subprocess.run(command, check=True)
        print(f"--- Training successful ---\nOutput directory: {config.output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"!!!!!! Training failed !!!!!!\nConfig: {config.output_dir}\nError: {e}")
        return False
    finally:
        if temp_yaml_path and os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)

def save_config_file(config: TrainingConfig, save_dir: str = "./configs") -> str:
    """Saves the configuration object to a YAML file."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, config.config_filename)
    params_dict = config.to_llamafactory_dict()
    
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(params_dict, f, sort_keys=False, allow_unicode=True)
    
    print(f"Configuration file saved to: {filepath}")
    return filepath