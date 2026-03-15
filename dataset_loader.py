import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import ast
import os

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=1024):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            seq,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float32)
        }

def load_and_split_data(csv_path, test_size=0.2, val_size=0.1, random_state=42):
    df = pd.read_csv(csv_path)
    
    # GO_IDs is stored as a string representation of a list in CSV
    df['GO_IDs'] = df['GO_IDs'].apply(ast.literal_eval)
    
    # Binarize labels
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(df['GO_IDs'])
    
    # For now, let's limit the number of labels to the top K most frequent if there are too many
    # However, let's see how many unique GO terms we have first
    num_unique_go = len(mlb.classes_)
    print(f"Total unique GO terms: {num_unique_go}")
    
    sequences = df['Sequence'].tolist()
    
    # Train/Val/Test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        sequences, labels, test_size=test_size, random_state=random_state
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size/(1-test_size), random_state=random_state
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), mlb

if __name__ == "__main__":
    csv_path = "data/protein_go_data.csv"
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found. Run data_collection.py first.")
    else:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), mlb = load_and_split_data(csv_path)
        print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
        
        # Load tokenizer
        model_checkpoint = "facebook/esm2_t6_8M_UR50D"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        
        # Create dataset
        train_ds = ProteinDataset(X_train[:10], y_train[:10], tokenizer)
        item = train_ds[0]
        print(f"Input IDs shape: {item['input_ids'].shape}")
        print(f"Labels shape: {item['labels'].shape}")
