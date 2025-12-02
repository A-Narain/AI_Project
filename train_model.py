# train_model.py - Training script for BERT Hate Speech Classifier

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
from tqdm import tqdm
import re

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

class HateSpeechDataset(Dataset):
    """Custom Dataset for hate speech data"""
    
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def preprocess_text(text):
    """Clean and preprocess tweet text"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = ' '.join(text.split())
    return text

def load_data(file_path):
    """Load and preprocess the dataset"""
    print("Loading dataset...")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Assuming columns: 'tweet' and 'class'
    # class: 0 = hate speech, 1 = offensive language, 2 = neither
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['class'].value_counts()}")
    
    # Preprocess texts
    df['tweet'] = df['tweet'].apply(preprocess_text)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        df['tweet'].values,
        df['class'].values,
        test_size=0.3,
        random_state=42,
        stratify=df['class']
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )
    
    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return predictions, true_labels, total_loss / len(dataloader)

def main():
    """Main training function"""
    
    # Load data (replace with your dataset path)
    data_path = 'hate_speech_dataset.csv'
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(data_path)
    
    # Initialize tokenizer and model
    print("\nInitializing model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3
    )
    model.to(device)
    
    # Create datasets
    train_dataset = HateSpeechDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset = HateSpeechDataset(X_val, y_val, tokenizer, MAX_LEN)
    test_dataset = HateSpeechDataset(X_test, y_test, tokenizer, MAX_LEN)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Validate
        val_preds, val_labels, val_loss = evaluate(model, val_loader, device)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Calculate metrics
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        print(f"Validation Macro F1: {val_f1:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print("Model saved!")
    
    # Load best model and evaluate on test set
    print("\n" + "="*50)
    print("Final Evaluation on Test Set")
    print("="*50)
    
    model.load_state_dict(torch.load('best_model.pt'))
    test_preds, test_labels, test_loss = evaluate(model, test_loader, device)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    
    # Calculate detailed metrics
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    print(f"Test Macro F1: {test_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        test_labels,
        test_preds,
        target_names=['Hate Speech', 'Offensive', 'Neither']
    ))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))
    
    # Save final model
    model.save_pretrained('hate_speech_model')
    tokenizer.save_pretrained('hate_speech_model')
    print("\nModel and tokenizer saved to 'hate_speech_model' directory")

if __name__ == '__main__':
    main()
