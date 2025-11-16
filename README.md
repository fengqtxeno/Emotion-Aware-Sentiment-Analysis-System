# 📊 Emotion-Aware Sentiment Analysis System

Advanced sentiment analysis system with social and emotional awareness capabilities, designed for regulatory and media monitoring applications.

## ✨ Core Capabilities

- **Social and Emotional Awareness**: Recognizes emotional cues, conversational norms, and cultural differences
- **Bias and Fairness Assessment**: Detects social biases and fairness issues in text
- **Cultural Sensitivity**: Evaluates content's impact on different cultural audiences
- **Emotional Safety**: Ensures responses adhere to safety guidelines
- **Human-AI Interaction Research**: Understands user-AI emotional interaction patterns

## 🚀 Quick Start

### 1. Install Dependencies
```bash
  pip install -r requirements.txt
# PyTorch GPU version (CUDA 12.8)
  pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```
### 2. Train Model
```bash
  python train_model.py --data_path datasets/market_sentiment.xlsx
```
### 3. Launch GUI Application
```bash
  python main.py
```
### 4. Command-line Analysis (Optional)
```bash
  python analyze_sentiment.py your_file.xlsx --output_dir results
```
## 📁 Project Structure
- **main.py**: GUI application for analysis
- **train_model.py**: Model training script (run once)
- **analyze_sentiment.py**: Command-line analysis tool
- **utils/**: Utility modules
- **data_loader.py**: Data loading and preprocessing
- **model_utils.py**: Model loading/saving/evaluation
- **social_awareness.py**: Social and emotional awareness core module
- **models/**: Trained models
- **datasets/**: Example dataset (replace with your data)
- **output/**: Analysis results are automatically saved here

## 💡 Usage Instructions
1. **Train Model**: First time only, train using annotated Excel data
1. **Analyze Content**:
   - Drag and drop Excel files into the GUI
   - View sentiment distribution, intensity analysis
   - Examine bias/fairness/cultural sensitivity scores
1. **Result Export**: Results are automatically saved with timestamped file names

## 🌐 Environment Compatibility
- GPU acceleration automatically enabled when available
- PyQt5 graphical interface (no browser required)
- English language support for all text processing

