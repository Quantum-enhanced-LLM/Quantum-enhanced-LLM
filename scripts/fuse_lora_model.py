import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

def fuse_model(base_model_path, lora_model_path, output_model_path):    
    # --- Load Base Model and Tokenizer ---
    print(f"Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,  # Choose dtype based on your model and hardware
        device_map="auto"           # Automatically select device (CPU/GPU)
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # --- Load LoRA Adapter ---
    print(f"Loading LoRA adapter from: {lora_model_path}")
    # Wrap the base model with PeftModel and load the LoRA weights
    model_with_lora = PeftModel.from_pretrained(base_model, lora_model_path)

    # --- Merge LoRA Weights into the Base Model ---
    print("Merging LoRA weights into the base model...")
    # This is the key step! It modifies the weights of the base model.
    # The `merge_and_unload()` method returns the original Transformers model with the LoRA weights merged.
    # For saving, we operate directly on this merged model.
    merged_model = model_with_lora.merge_and_unload()

    # Ensure the tokenizer has a pad_token, as some models may not.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Or another suitable pad token
        
    output_model_path = Path(output_model_path)
    # --- Save the Merged Model ---
    print(f"Saving merged model to: {output_model_path}")
    merged_model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)

    print("Model merging complete and saved successfully!")
    print(f"You can now load your new model with: AutoModelForCausalLM.from_pretrained('{output_model_path}')")

if __name__ == '__main__':
    # --- Configuration Parameters ---
    # Path to your original base model
    base_model_path = "/data3/Share_Zone/models/Qwen3-1.7B"
    
    # Base path to your trained LoRA models
    # e.g., "./saves/chatglm3-6b/lora/your_lora_exp_name"
    lora_model_basepath = "/data2/home/agony/qpeft_large_epoch_test/qpeft/QPEFT_Large_Epoch_Test/lora/Qwen3-1.7B/"    
    lora_model_basepath = Path(lora_model_basepath)

    # Base path where you want to save the newly merged models
    # e.g., "./merged_models/chatglm3-6b-with-your-lora"
    output_model_basepath = Path.cwd() 

    for lora_rank in [4, 6, 8, 10]:
        for dataset in ['CPsyCounD_eval', 'GSM8K_MAIN_TRAIN']:
            lora_model_path = lora_model_basepath / dataset / "samples_3000" / f'lora_r{lora_rank}_q_proj_v_proj'
            output_model_path = output_model_basepath / dataset / f'lora_r{lora_rank}'

            if not lora_model_path.exists():
                print(f"Skipping, LoRA path not found: {lora_model_path}")
                continue

            fuse_model(base_model_path, lora_model_path, output_model_path)