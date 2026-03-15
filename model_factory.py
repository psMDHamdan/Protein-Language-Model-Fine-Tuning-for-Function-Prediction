from transformers import EsmForSequenceClassification, EsmConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch

def create_model(model_checkpoint, num_labels, use_lora=True):
    """
    Creates an ESM model with a classification head and optional LoRA adapters.
    """
    print(f"Loading model checkpoint: {model_checkpoint}")
    
    config = EsmConfig.from_pretrained(model_checkpoint)
    config.num_labels = num_labels
    config.problem_type = "multi_label_classification"
    
    model = EsmForSequenceClassification.from_pretrained(
        model_checkpoint,
        config=config
    )
    
    if use_lora:
        print("Applying LoRA adapters...")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"] # Standard for ESM-2
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model

if __name__ == "__main__":
    model_checkpoint = "facebook/esm2_t6_8M_UR50D"
    num_labels = 4014 # Based on dataset_loader.py output
    
    model = create_model(model_checkpoint, num_labels)
    print(f"Model created. Number of labels: {num_labels}")
    
    # Check output shape
    dummy_input = torch.randint(0, 30, (1, 1024))
    output = model(dummy_input)
    print(f"Output logits shape: {output.logits.shape}")
