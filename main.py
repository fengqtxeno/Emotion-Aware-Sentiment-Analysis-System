import os
import sys
import pandas as pd
import torch
import matplotlib

matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QTextEdit,
    QVBoxLayout, QWidget, QFileDialog, QMessageBox, QHBoxLayout,
    QTabWidget, QTableWidget, QTableWidgetItem, QSplitter, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QIcon, QPixmap, QPalette, QColor

from utils.data_loader import load_and_preprocess_data, clean_text
from utils.model_utils import load_model_and_tokenizer, predict_sentiment, generate_sentiment_distribution_plot
from utils.social_awareness import SocialAwarenessAnalyzer

# Global configuration
plt.rcParams['font.sans-serif'] = ['Arial']  # Standard font
plt.rcParams['axes.unicode_minus'] = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "models/sentiment_model_v1"
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


class AnalysisWorker(QThread):
    """Background analysis thread"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(str, str, str)
    error = pyqtSignal(str)

    def __init__(self, file_path, output_dir=None, model_path=MODEL_PATH):
        super().__init__()
        self.file_path = file_path
        self.output_dir = output_dir
        self.model_path = model_path

    def run(self):
        """Execute analysis task"""
        try:
            # Create output directory
            if self.output_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.output_dir = Path("output") / timestamp
            self.output_dir = Path(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Load model
            self.progress.emit("Loading sentiment analysis model...")
            model, tokenizer, device = load_model_and_tokenizer(
                self.model_path, num_labels=3, device=DEVICE
            )

            # Load social awareness analyzer
            self.progress.emit("Initializing social awareness module...")
            social_analyzer = SocialAwarenessAnalyzer()

            # Load data
            self.progress.emit("Loading Excel data...")
            data = load_and_preprocess_data(self.file_path)
            df_raw = data['full']
            label_map = data['label_map']

            self.progress.emit(f"Starting analysis of {len(df_raw)} records...")

            # Analyze each record
            results = []
            total = len(df_raw)

            for idx, row in df_raw.iterrows():
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

                # Update progress
                if (idx + 1) % 10 == 0 or idx == total - 1:
                    self.progress.emit(f"Analyzed {idx + 1}/{total} records ({(idx + 1) / total * 100:.1f}%)")

            # Save results
            results_df = pd.DataFrame(results)
            excel_path = self.output_dir / f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            results_df.to_excel(excel_path, index=False)

            # Generate visualizations
            image_path = self.output_dir / f"sentiment_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            generate_sentiment_distribution_plot(results_df, str(image_path))

            # Generate summary
            summary = self.generate_analysis_summary(results_df)
            summary_path = self.output_dir / f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)

            self.finished.emit(str(excel_path), str(image_path), str(summary_path))

        except Exception as e:
            import traceback
            error_msg = f"Analysis failed: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)

    def generate_analysis_summary(self, df):
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


class SentimentDashboard(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotion-Aware Sentiment Analysis System")
        self.resize(1200, 800)

        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            QMessageBox.warning(
                self, "Warning",
                "Sentiment analysis model not found. Please run train_model.py first!"
            )

        # Main interface
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Top title
        title_label = QLabel("Emotion-Aware Sentiment Analysis System")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; padding: 10px; background-color: #f8f9fa;")
        main_layout.addWidget(title_label)

        # Middle area - split into left and right panels
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - operation panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Drag-and-drop area
        self.drop_area = QLabel("Drag and drop Excel file here\nor click button below to select file")
        self.drop_area.setAlignment(Qt.AlignCenter)
        self.drop_area.setAcceptDrops(True)
        self.drop_area.setStyleSheet("""
            border: 2px dashed #3498db;
            border-radius: 10px;
            padding: 40px;
            background-color: #f8f9fa;
            color: #7f8c8d;
            font-size: 16px;
        """)
        left_layout.addWidget(self.drop_area)

        # Button area
        btn_layout = QHBoxLayout()

        self.select_btn = QPushButton("Select File")
        self.select_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.select_btn.clicked.connect(self.select_file)
        btn_layout.addWidget(self.select_btn)

        self.open_folder_btn = QPushButton("Open Output Folder")
        self.open_folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self.open_output_folder)
        btn_layout.addWidget(self.open_folder_btn)

        left_layout.addLayout(btn_layout)

        # Log area
        left_layout.addWidget(QLabel("Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                font-family: Consolas, Monaco, monospace;
                font-size: 12px;
                background-color: #f8f9fa;
            }
        """)
        left_layout.addWidget(self.log_text)

        # Right panel - results panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Tab widget
        self.tab_widget = QTabWidget()

        # Summary tab
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 15px;
                background-color: #f8f9fa;
                font-family: Arial;
            }
        """)
        summary_layout.addWidget(self.summary_text)

        # Results table tab
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(9)
        self.results_table.setHorizontalHeaderLabels([
            "RID", "Title", "Sentiment", "Confidence", "Intensity",
            "Bias Score", "Sensitivity", "Safety", "Details"
        ])
        self.results_table.setSortingEnabled(True)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.results_table.cellDoubleClicked.connect(self.show_detail)
        results_layout.addWidget(self.results_table)

        # Social awareness tab
        social_tab = QWidget()
        social_layout = QVBoxLayout(social_tab)

        self.social_text = QTextEdit()
        self.social_text.setReadOnly(True)
        self.social_text.setStyleSheet("""
            QTextEdit {
                padding: 15px;
                font-family: Arial;
                line-height: 1.5;
            }
        """)
        # Set social awareness description
        social_content = """
🤝 Social and Emotional Awareness Analysis
=====================

This system is based on Leiden University's emotionally and socially aware NLP research, providing multi-dimensional analysis:

1. 👥 Social Bias Detection
   • Identifies gender, regional, economic status, age, and professional bias expressions
   • Quantifies bias density to flag potential discriminatory content
   • Example: "These outsiders always..." may contain regional bias

2. 🌍 Cultural Sensitivity Assessment
   • Monitors ethnic, religious, political, and historical sensitive content
   • Evaluates potential impact on audiences from different cultural backgrounds
   • Example: Respectful treatment of religious customs

3. ⚠️ Emotional Safety
   • Detects violence, discrimination, and illegal content terminology
   • Assesses emotional health and constructiveness of expression
   • Example: Avoiding inflammatory language and personal attacks

4. 💬 Interaction Pattern Analysis
   • Identifies questioning, commanding, emotional expression, and factual statement patterns
   • Evaluates emotional consistency (alignment between emotional words and overall sentiment)
   • Example: Surface questioning with hidden strong negative emotions

5. 📊 Sentiment Intensity Quantification
   • Quantifies sentiment from -1.0 (strongly negative) to +1.0 (strongly positive)
   • Identifies emotional polarization for fine-grained analysis

This multi-dimensional analysis enables regulatory departments to:
• See not just "what sentiment was expressed" but "how it was expressed" and "to whom it was addressed"
• Identify potential public opinion risk points beyond surface negative content
• Understand different group reactions to the same event
• Develop more targeted communication and guidance strategies

Design principle: Enhance human judgment rather than replace human review.
        """
        self.social_text.setPlainText(social_content)
        social_layout.addWidget(self.social_text)

        # Add tabs
        self.tab_widget.addTab(summary_tab, "Analysis Summary")
        self.tab_widget.addTab(results_tab, "Detailed Results")
        self.tab_widget.addTab(social_tab, "Social Awareness")

        right_layout.addWidget(self.tab_widget)

        # Add to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])

        main_layout.addWidget(splitter)

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage(f"Device: {DEVICE} | Model: {MODEL_PATH}")

        # Set style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ecf0f1;
            }
            QTabWidget::pane {
                border: 1px solid #ddd;
                border-radius: 4px;
                background: white;
            }
            QTabBar::tab {
                background: #f8f9fa;
                border: 1px solid #ddd;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #3498db;
                color: white;
            }
            QTableWidget {
                border: 1px solid #ddd;
                gridline-color: #eee;
            }
            QTableWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
        """)

        self.current_file = None
        self.output_dir = None
        self.results_df = None

    def dragEnterEvent(self, event):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Handle drop event"""
        urls = event.mimeData().urls()
        if urls and urls[0].toLocalFile().endswith('.xlsx'):
            self.current_file = urls[0].toLocalFile()
            self.drop_area.setText(f"Selected file:\n{self.current_file}")
            self.start_analysis()

    def select_file(self):
        """Select file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Excel File", "", "Excel Files (*.xlsx)"
        )
        if file_path:
            self.current_file = file_path
            self.drop_area.setText(f"Selected file:\n{self.current_file}")
            self.start_analysis()

    def start_analysis(self):
        """Start analysis"""
        if not self.current_file:
            return

        if not os.path.exists(MODEL_PATH):
            QMessageBox.warning(
                self, "Error",
                "Model not found. Please run train_model.py first!"
            )
            return

        self.log_text.clear()
        self.log_text.append(f"Starting analysis of file: {self.current_file}")
        self.log_text.append(f"Using device: {DEVICE}")
        self.log_text.append("Initializing analysis thread...")

        self.worker = AnalysisWorker(self.current_file, model_path=MODEL_PATH)
        self.worker.progress.connect(self.update_log)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

        self.select_btn.setEnabled(False)

    def update_log(self, message):
        """Update log"""
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def on_analysis_finished(self, excel_path, image_path, summary_path):
        """Analysis finished callback"""
        self.update_log("✅ Analysis complete!")
        self.update_log(f"Results saved to: {excel_path}")
        self.update_log(f"Visualization: {image_path}")
        self.update_log(f"Summary: {summary_path}")

        self.output_dir = Path(excel_path).parent
        self.open_folder_btn.setEnabled(True)
        self.select_btn.setEnabled(True)

        # Load results data
        try:
            self.results_df = pd.read_excel(excel_path)
            self.display_results(self.results_df)
            self.generate_summary(self.results_df)
        except Exception as e:
            self.update_log(f"⚠️ Error loading results data: {str(e)}")

    def on_analysis_error(self, error_msg):
        """Analysis error callback"""
        self.update_log(f"❌ Analysis failed: {error_msg}")
        QMessageBox.critical(self, "Error", f"Analysis failed:\n{error_msg}")
        self.select_btn.setEnabled(True)

    def open_output_folder(self):
        """Open output folder"""
        if self.output_dir and self.output_dir.exists():
            try:
                os.startfile(str(self.output_dir))
            except Exception as e:
                self.update_log(f"⚠️ Unable to open folder: {str(e)}")

    def display_results(self, df):
        """Display results in table"""
        self.results_table.setRowCount(len(df))

        for row_idx, (_, row) in enumerate(df.iterrows()):
            # RID
            rid_item = QTableWidgetItem(str(row['RID']))
            self.results_table.setItem(row_idx, 0, rid_item)

            # Title (truncated)
            title = str(row['title']) if pd.notna(row['title']) else ""
            title_display = title[:50] + "..." if len(title) > 50 else title
            title_item = QTableWidgetItem(title_display)
            self.results_table.setItem(row_idx, 1, title_item)

            # Sentiment
            sentiment = str(row['predicted_sentiment'])
            sentiment_item = QTableWidgetItem(sentiment)

            # Set color
            if sentiment == "negative":
                sentiment_item.setBackground(QColor(255, 204, 204))  # Light red
            elif sentiment == "positive":
                sentiment_item.setBackground(QColor(204, 255, 204))  # Light green
            else:
                sentiment_item.setBackground(QColor(255, 255, 204))  # Light yellow

            self.results_table.setItem(row_idx, 2, sentiment_item)

            # Confidence
            confidence = row['sentiment_confidence']
            conf_item = QTableWidgetItem(f"{confidence:.2%}")
            self.results_table.setItem(row_idx, 3, conf_item)

            # Sentiment intensity
            score = row['sentiment_score']
            score_item = QTableWidgetItem(f"{score:.3f}")
            self.results_table.setItem(row_idx, 4, score_item)

            # Bias score
            bias_score = row['bias_score']
            bias_item = QTableWidgetItem(f"{bias_score:.3f}")
            self.results_table.setItem(row_idx, 5, bias_item)

            # Sensitivity
            sensitivity_score = row['cultural_sensitivity_score']
            sensitivity_item = QTableWidgetItem(f"{sensitivity_score:.3f}")
            self.results_table.setItem(row_idx, 6, sensitivity_item)

            # Safety
            safety_score = row['safety_score']
            safety_item = QTableWidgetItem(f"{safety_score:.3f}")
            self.results_table.setItem(row_idx, 7, safety_item)

            # Details button
            detail_btn = QPushButton("View")
            detail_btn.clicked.connect(lambda _, r=row_idx: self.show_detail(r, 0))
            self.results_table.setCellWidget(row_idx, 8, detail_btn)

        self.results_table.resizeColumnsToContents()

    def show_detail(self, row, col):
        """Show detailed information"""
        if self.results_df is None:
            return

        row_data = self.results_df.iloc[row]
        title = str(row_data['title']) if pd.notna(row_data['title']) else ""
        body = str(row_data['body']) if pd.notna(row_data['body']) else ""

        detail_text = f"""
📊 Detailed Analysis Report
================

📌 Title:
{title}

📝 Body:
{body}

🏷️ Sentiment Analysis:
- Sentiment Category: {row_data['predicted_sentiment']}
- Confidence: {row_data['sentiment_confidence']:.2%}
- Sentiment Intensity: {row_data['sentiment_score']:.3f} ({'negative-leaning' if row_data['sentiment_score'] < 0 else 'positive-leaning'})

🎯 Social Awareness:
- Social Bias Score: {row_data['bias_score']:.3f} ({'high' if row_data['bias_score'] > 0.05 else 'medium' if row_data['bias_score'] > 0.02 else 'low'})
- Cultural Sensitivity: {row_data['cultural_sensitivity_score']:.3f} ({'highly sensitive' if row_data['cultural_sensitivity_score'] > 0.7 else 'moderately sensitive' if row_data['cultural_sensitivity_score'] > 0.4 else 'low sensitivity'})
- Emotional Safety: {row_data['safety_score']:.3f} ({'safe' if row_data['safety_score'] > 0.8 else 'moderate' if row_data['safety_score'] > 0.5 else 'requires attention'})
        """

        QMessageBox.information(self, "Detailed Analysis", detail_text)

    def generate_summary(self, df):
        """Generate analysis summary"""
        total = len(df)
        sentiment_counts = df['predicted_sentiment'].value_counts()
        avg_sentiment = df['sentiment_score'].mean()
        avg_bias = df['bias_score'].mean()
        avg_safety = df['safety_score'].mean()

        summary = f"""
📊 Sentiment Analysis Summary
================
Total records analyzed: {total}
Sentiment distribution:
  - Negative: {sentiment_counts.get('negative', 0)} ({sentiment_counts.get('negative', 0) / total * 100:.1f}%)
  - Neutral: {sentiment_counts.get('neutral', 0)} ({sentiment_counts.get('neutral', 0) / total * 100:.1f}%)
  - Positive: {sentiment_counts.get('positive', 0)} ({sentiment_counts.get('positive', 0) / total * 100:.1f}%)

📈 Key Metrics
- Average sentiment intensity: {avg_sentiment:.3f} ({'negative-leaning' if avg_sentiment < 0 else 'positive-leaning'})
- Average social bias score: {avg_bias:.3f} ({'high' if avg_bias > 0.05 else 'medium' if avg_bias > 0.02 else 'low'})
- Average emotional safety: {avg_safety:.3f} ({'safe' if avg_safety > 0.8 else 'moderate' if avg_safety > 0.5 else 'requires attention'})

🔍 Insights & Recommendations
- Sentiment polarization index: {abs(df['sentiment_score']).mean():.2f}
- High bias content ratio: {(df['bias_score'] > 0.05).sum()}/{total} ({(df['bias_score'] > 0.05).sum() / total * 100:.1f}%)
- Low safety content ratio: {(df['safety_score'] < 0.5).sum()}/{total} ({(df['safety_score'] < 0.5).sum() / total * 100:.1f}%)

💡 Strategy Recommendations
1. For negative sentiment content, establish rapid response mechanisms
2. Prioritize review of high bias content to prevent propagation of biases
3. Handle low safety content with priority to prevent public opinion escalation
4. Leverage neutral content as communication bridges to guide rational discussion
        """

        self.summary_text.setPlainText(summary)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    # Set palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(236, 240, 241))
    palette.setColor(QPalette.WindowText, QColor(44, 62, 80))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(248, 249, 250))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 224))
    palette.setColor(QPalette.ToolTipText, QColor(44, 62, 80))
    palette.setColor(QPalette.Text, QColor(44, 62, 80))
    palette.setColor(QPalette.Button, QColor(236, 240, 241))
    palette.setColor(QPalette.ButtonText, QColor(44, 62, 80))
    palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
    palette.setColor(QPalette.Highlight, QColor(52, 152, 219))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = SentimentDashboard()
    window.show()
    sys.exit(app.exec_())