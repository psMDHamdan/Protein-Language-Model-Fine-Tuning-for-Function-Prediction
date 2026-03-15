import os
import numpy as np
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import f1_score, precision_score, recall_score
from dataset_loader import load_and_split_data, ProteinDataset
from model_factory import create_model

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Apply sigmoid to logits to get probabilities
    probs = 1 / (1 + np.exp(-logits))
    # Threshold at 0.5
    predictions = (probs > 0.5).astype(int)
    
    # Calculate metrics
    # Using weighted average for multi-label F1
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(
    csv_path="data/protein_go_data.csv",
    model_checkpoint="facebook/esm2_t6_8M_UR50D",
    output_dir="outputs/esm2_go_finetuned",
    num_train_epochs=3,
    batch_size=8,
    learning_rate=2e-4,
    train_data=None,
    val_data=None,
    mlb=None
):
    # 1. Load data if not provided
    if train_data is None or val_data is None or mlb is None:
        (X_train, y_train), (X_val, y_val), (_, _), mlb = load_and_split_data(csv_path)
    else:
        X_train, y_train = train_data
        X_val, y_val = val_data
        
    num_labels = len(mlb.classes_)
    
    # 2. Initialize Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = create_model(model_checkpoint, num_labels, use_lora=True)
    
    # 3. Create Datasets
    # For initial testing, let's use a smaller subset if needed, 
    # but 3500 is small enough to try on CPU/small GPU.
    train_ds = ProteinDataset(X_train, y_train, tokenizer)
    val_ds = ProteinDataset(X_val, y_val, tokenizer)
    
    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        remove_unused_columns=False, # Important for ESM models in some versions
        push_to_hub=False,
        report_to="none"
    )
    
    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    
    # 6. Train
    print("Starting training...")
    trainer.train()
    
    # 7. Save the final model
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    # Also save the MultiLabelBinarizer classes for later use in prediction
    import pickle
    with open(os.path.join(output_dir, "mlb.pkl"), "wb") as f:
        pickle.dump(mlb, f)
    
    return trainer, mlb

if __name__ == "__main__":
    # Smoke test on very small subset
    if not os.path.exists("data/protein_go_data.csv"):
        print("Run data_collection.py first.")
    else:
        print("Running smoke test training on a very small subset...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test), mlb = load_and_split_data("data/protein_go_data.csv")
        
        train_model(
            num_train_epochs=1, 
            batch_size=2,
            train_data=(X_train[:10], y_train[:10]),
            val_data=(X_val[:5], y_val[:5]),
            mlb=mlb
        )
