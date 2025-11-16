import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime

from utils.data_loader import load_and_preprocess_data, SocialMediaDataset
from utils.model_utils import save_model, evaluate_model, plot_training_history

# Global configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRE_TRAINED_MODEL_NAME = 'hfl/chinese-roberta-wwm-ext'
NUM_LABELS = 3
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 2e-5
MAX_LEN = 512
MODEL_SAVE_PATH = "models/sentiment_model_v1"


def train_epoch(model, data_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    losses = []

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return np.mean(losses)


def train_sentiment_model(data_path, model_save_path=MODEL_SAVE_PATH, epochs=EPOCHS):
    """Train sentiment analysis model"""
    print(f"Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load and preprocess data
    print("Loading Excel data...")
    data = load_and_preprocess_data(data_path)
    df_train = data['train']
    df_val = data['val']
    label_map = data['label_map']

    print(f"Loaded {len(data['full'])} annotated records with distribution:")
    print(data['full']['label'].value_counts().rename(label_map))
    print(f"Training set: {len(df_train)} records, Validation set: {len(df_val)} records")

    # Load model and tokenizer
    print("Loading pre-trained model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        PRE_TRAINED_MODEL_NAME,
        num_labels=NUM_LABELS
    ).to(DEVICE)

    # Prepare datasets
    train_dataset = SocialMediaDataset(df_train['text'].values, df_train['label'].values, tokenizer, MAX_LEN)
    val_dataset = SocialMediaDataset(df_val['text'].values, df_val['label'].values, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE) if len(df_val) > 0 else None

    # Train model
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    best_f1 = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    for epoch in range(epochs):
        print(f'--- Epoch {epoch + 1}/{epochs} ---')
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        history['train_loss'].append(train_loss)
        print(f"Training loss: {train_loss:.4f}")

        # Evaluate
        if val_loader and len(df_val) > 0:
            report, _, _ = evaluate_model(model, val_loader, DEVICE, label_map)
            print("\nEvaluation Report:")
            print(report)

            # Extract weighted F1 score
            lines = report.split('\n')
            f1_line = [line for line in lines if 'weighted avg' in line]
            if f1_line:
                weighted_f1 = float(f1_line[0].split()[-2])
                history['val_f1'].append(weighted_f1)

                # Save best model
                if weighted_f1 > best_f1:
                    best_f1 = weighted_f1
                    print(f"New best model (F1: {best_f1:.4f}), saving to {model_save_path}")
                    save_model(model, tokenizer, model_save_path)
            else:
                print("Could not extract F1 score, skipping model save")
        else:
            # Save model every epoch if no validation set
            print(f"Saving model to {model_save_path}")
            save_model(model, tokenizer, model_save_path)

    # Save training history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = os.path.join(model_save_path, f'training_history_{timestamp}.png')
    plot_training_history(history, history_path)

    print(f"Training complete! Best model F1: {best_f1:.4f}, saved to {model_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    parser.add_argument('--data_path', type=str, default='datasets/market_sentiment.xlsx',
                        help='Path to Excel data file')
    parser.add_argument('--model_save_path', type=str, default=MODEL_SAVE_PATH,
                        help='Model save path')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of training epochs')

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs('datasets', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    train_sentiment_model(args.data_path, args.model_save_path, args.epochs)