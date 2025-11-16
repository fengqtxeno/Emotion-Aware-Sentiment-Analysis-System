import os
import pandas as pd
import torch
import matplotlib

matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from pathlib import Path
import argparse

from utils.data_loader import load_and_preprocess_data, clean_text
from utils.model_utils import load_model_and_tokenizer, predict_sentiment, generate_sentiment_distribution_plot
from utils.social_awareness import SocialAwarenessAnalyzer

# Global configuration
plt.rcParams['font.sans-serif'] = ['Arial']  # Standard English font
plt.rcParams['axes.unicode_minus'] = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "models/sentiment_model_v1"
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def analyze_sentiment_batch(input_file, output_dir=None, model_path=MODEL_PATH):
    """Batch sentiment analysis"""
    print(f"Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output") / timestamp
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading sentiment analysis model...")
    model, tokenizer, device = load_model_and_tokenizer(model_path, num_labels=3, device=DEVICE)

    # Load social awareness analyzer
    print("Initializing social awareness module...")
    social_analyzer = SocialAwarenessAnalyzer()

    # Load data
    print("Loading Excel data...")
    data = load_and_preprocess_data(input_file)
    df_raw = data['full']
    label_map = data['label_map']

    print(f"Starting analysis of {len(df_raw)} records...")

    # Analyze each record
    results = []
    for idx, row in tqdm(df_raw.iterrows(), total=len(df_raw)):
        title = str(row['title']) if pd.notna(row['title']) else ""
        body = str(row['body']) if pd.notna(row['body']) else ""
        full_text = clean_text((title + ". " + body) if body.strip() else title)

        # Ensure content exists
        if not full_text.strip():
            full_text = "(No content)"
            sentiment_label = "neutral"
            sentiment_confidence = 1.0
            sentiment_score = 0.0
            social_bias = {}
            cultural_sensitivity = {}
            emotional_safety = {}
        else:
            # Sentiment analysis
            predictions = predict_sentiment(
                model, tokenizer, [full_text], device, max_len=512, batch_size=1
            )[0]

            pred_label_id = predictions['predicted_label_id']
            confidence = predictions['confidence']
            sentiment_score = predictions['sentiment_score']
            probs = predictions['probabilities']

            sentiment_label = label_map[pred_label_id]

            # Social awareness analysis
            social_bias = social_analyzer.analyze_social_bias(full_text)
            cultural_sensitivity = social_analyzer.analyze_cultural_sensitivity(full_text)
            emotional_safety = social_analyzer.analyze_emotional_safety(full_text)
            emotional_cues = social_analyzer.analyze_emotional_cues(full_text, probs)

        # Prepare results
        result = {
            "RID": row['RID'] if 'RID' in row and pd.notna(row['RID']) else idx,
            "title": title,
            "body": body,
            "predicted_sentiment": sentiment_label,
            "sentiment_confidence": confidence,
            "sentiment_score": sentiment_score
        }

        # Add social awareness results
        result.update({
            "bias_score": social_bias.get('overall_bias_score', 0),
            "cultural_sensitivity_score": cultural_sensitivity.get('score', 0),
            "safety_score": emotional_safety.get('score', 1.0)
        })

        results.append(result)

    # Save results
    results_df = pd.DataFrame(results)
    excel_path = output_dir / f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    results_df.to_excel(excel_path, index=False)
    print(f"Analysis results saved to: {excel_path}")

    # Generate visualizations
    image_path = output_dir / f"sentiment_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    generate_sentiment_distribution_plot(results_df, image_path)

    # Generate summary
    summary = generate_analysis_summary(results_df)
    summary_path = output_dir / f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"Analysis summary saved to: {summary_path}")

    print("Analysis complete!")
    return excel_path, image_path, summary_path


def generate_analysis_summary(df):
    """Generate analysis summary"""
    total = len(df)
    sentiment_counts = df['predicted_sentiment'].value_counts()
    avg_sentiment = df['sentiment_score'].mean()
    avg_bias = df['bias_score'].mean()
    avg_safety = df['safety_score'].mean()

    high_bias_ratio = (df['bias_score'] > 0.05).sum() / total
    low_safety_ratio = (df['safety_score'] < 0.5).sum() / total

    summary = f"""
📊 Sentiment Analysis Summary
===========================
Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total records analyzed: {total}
Sentiment distribution:
  - Negative: {sentiment_counts.get('negative', 0)} ({sentiment_counts.get('negative', 0) / total * 100:.1f}%)
  - Neutral: {sentiment_counts.get('neutral', 0)} ({sentiment_counts.get('neutral', 0) / total * 100:.1f}%)
  - Positive: {sentiment_counts.get('positive', 0)} ({sentiment_counts.get('positive', 0) / total * 100:.1f}%)

📊 Key Metrics:
- Average sentiment intensity: {avg_sentiment:.3f} ({'negative-leaning' if avg_sentiment < 0 else 'positive-leaning'})
- Average social bias score: {avg_bias:.3f} ({'high' if avg_bias > 0.05 else 'medium' if avg_bias > 0.02 else 'low'})
- Average emotional safety: {avg_safety:.3f} ({'safe' if avg_safety > 0.8 else 'moderate' if avg_safety > 0.5 else 'requires attention'})

🔍 Key Insights:
- Sentiment polarization index: {abs(df['sentiment_score']).mean():.2f} (0-1, higher indicates stronger emotions)
- High bias content ratio: {high_bias_ratio:.1%}
- Low safety content ratio: {low_safety_ratio:.1%}

💡 Recommendations:
1. Monitor negative sentiment topics for timely public response
2. Conduct human review of high-bias content to prevent bias propagation
3. Prioritize handling of low safety content to prevent escalation
4. Use neutral content as communication bridges to guide rational discussion
    """

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch sentiment analysis tool')
    parser.add_argument('input_file', type=str, help='Input Excel file path')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Model path')

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist! Train the model first using train_model.py.")
        exit(1)

    analyze_sentiment_batch(args.input_file, args.output_dir, args.model_path)