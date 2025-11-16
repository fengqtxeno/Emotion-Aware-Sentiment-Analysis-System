import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Dataset
import torch


def clean_text(text):
    """Clean text by removing HTML tags, links, and extra whitespace"""
    if not isinstance(text, str):
        return ""
    # Remove image tags
    text = re.sub(r'<IMAGE SRC="[^"]*">', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Consolidate extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_preprocess_data(file_path, test_size=0.2, random_state=42):
    """Load and preprocess Excel data, returning training set, validation set, and full dataset"""
    # Read Excel file
    df = pd.read_excel(file_path, sheet_name="Sheet0")

    # Keep rows with content and sentiment labels
    df_clean = df.dropna(subset=['body', 'sentiment']).copy()

    # Create label mapping
    label_to_id = {"negative": 0, "neutral": 1, "positive": 2}
    id_to_label = {v: k for k, v in label_to_id.items()}

    # Map labels
    df_clean['label'] = df_clean['sentiment'].map(label_to_id)
    df_clean = df_clean[df_clean['label'].notna()].copy()
    df_clean['label'] = df_clean['label'].astype(int)

    # Clean and concatenate text
    df_clean['text'] = (
            df_clean['title'].fillna("").astype(str) + ". " +
            df_clean['body'].fillna("").astype(str)
    ).apply(clean_text)

    # Handle class imbalance
    if len(df_clean) > 0:
        ros = RandomOverSampler(random_state=random_state)
        texts = df_clean['text'].values.reshape(-1, 1)
        labels = df_clean['label'].values

        texts_res, labels_res = ros.fit_resample(texts, labels)
        df_balanced = pd.DataFrame({
            'text': texts_res.flatten(),
            'label': labels_res
        })
    else:
        df_balanced = df_clean.copy()

    # Split dataset
    if len(df_balanced) > 1:
        df_train, df_val = train_test_split(
            df_balanced,
            test_size=test_size,
            random_state=random_state,
            stratify=df_balanced['label']
        )
    else:
        df_train = df_balanced
        df_val = pd.DataFrame(columns=df_balanced.columns)

    return {
        'train': df_train,
        'val': df_val,
        'full': df_clean,
        'label_map': id_to_label,
        'label_to_id': label_to_id
    }


class SocialMediaDataset(Dataset):
    """Dataset class for model training"""

    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }