import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm


def load_model_and_tokenizer(model_path, num_labels=3, device=None):
    """Load saved model and tokenizer"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels
        ).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to reload model with mismatched size handling...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        ).to(device)

    return model, tokenizer, device


def save_model(model, tokenizer, save_path):
    """Save model and tokenizer to specified path"""
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to: {save_path}")


def evaluate_model(model, data_loader, device, label_map=None):
    """Evaluate model performance"""
    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    # Generate classification report
    if label_map and len(set(actuals)) == len(label_map):
        target_names = [label_map[i] for i in range(len(label_map))]
        report = classification_report(
            actuals,
            predictions,
            target_names=target_names,
            zero_division=0
        )
    else:
        report = classification_report(actuals, predictions, zero_division=0)

    return report, predictions, actuals


def predict_sentiment(model, tokenizer, texts, device, max_len=512, batch_size=16):
    """Predict sentiment for a list of texts"""
    model.eval()
    results = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Encode
        encodings = tokenizer(
            batch_texts,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(**encodings)

        # Calculate probabilities and predictions
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = torch.argmax(outputs.logits, dim=1)

        # Calculate sentiment scores
        for j in range(len(batch_texts)):
            pred_id = predictions[j].item()
            confidence = probs[j][pred_id].item()
            sentiment_score = (-1) * probs[j][0].item() + 0 * probs[j][1].item() + 1 * probs[j][2].item()

            results.append({
                'predicted_label_id': pred_id,
                'confidence': confidence,
                'sentiment_score': sentiment_score,
                'probabilities': probs[j].cpu().numpy()
            })

    return results


def plot_training_history(history, save_path=None):
    """Plot training history chart"""
    plt.figure(figsize=(12, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy curve (if available)
    if 'train_acc' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        if 'val_acc' in history:
            plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training history chart saved to: {save_path}")
    plt.close()


def generate_sentiment_distribution_plot(results_df, save_path=None):
    """Generate sentiment distribution visualization"""
    plt.figure(figsize=(15, 12))

    # 1. Sentiment distribution
    plt.subplot(2, 2, 1)
    sentiment_counts = results_df['predicted_sentiment'].value_counts()
    colors = sns.color_palette("Set2", len(sentiment_counts))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
    plt.title('Sentiment Distribution', fontsize=14)

    # 2. Sentiment intensity distribution
    plt.subplot(2, 2, 2)
    sns.histplot(results_df['sentiment_score'], bins=20, kde=True, color='skyblue')
    plt.title('Sentiment Intensity Distribution (-1=negative, +1=positive)', fontsize=14)
    plt.axvline(0, color='red', linestyle='--', label='Neutral line')
    plt.xlabel('Sentiment Intensity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()

    # 3. Bias distribution
    if 'bias_score' in results_df.columns:
        plt.subplot(2, 2, 3)
        sns.histplot(results_df['bias_score'], bins=20, kde=True, color='salmon')
        plt.title('Social Bias Level Distribution', fontsize=14)
        plt.xlabel('Bias Density', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)

    # 4. Safety distribution
    if 'safety_score' in results_df.columns:
        plt.subplot(2, 2, 4)
        sns.histplot(results_df['safety_score'], bins=20, kde=True, color='lightgreen')
        plt.title('Emotional Safety Distribution (1=safest)', fontsize=14)
        plt.xlabel('Safety Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Sentiment distribution chart saved to: {save_path}")
    plt.close()