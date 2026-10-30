import os
import torch
import torch.nn.functional as F
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report

import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from level_1_ml.train import generate_synthetic_data, TextPreprocessor, extract_lexicon_features
from level_2_dl.train_dl import Vocabulary, SentimentLSTM

def main():
    print("Loading test dataset for ML vs DL benchmarking...")
    # Use larger dataset matching training
    df = generate_synthetic_data(1500)
    preprocessor = TextPreprocessor()
    df['cleaned_headline'] = df['headline'].apply(preprocessor.preprocess)
    
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    y = df['sentiment'].map(sentiment_map).values
    
    # Chronological Split (same as training splits)
    split_idx = int(len(df) * 0.8)
    test_texts = df['cleaned_headline'].iloc[split_idx:].values
    y_test = y[split_idx:]
    
    # ------------------ EVALUATE LEVEL 1 ML MODEL ------------------
    print("\n--- Evaluating Level 1 ML Baseline ---")
    ml_model_path = os.path.join(ROOT_DIR, 'models', 'best_sentiment_model.joblib')
    tfidf_path = os.path.join(ROOT_DIR, 'models', 'tfidf_vectorizer.joblib')
    
    if not os.path.exists(ml_model_path) or not os.path.exists(tfidf_path):
        print("ML Baseline files missing. Make sure train.py has run successfully first.")
        # Create models folder and run training if missing
        import subprocess
        subprocess.run([sys.executable, os.path.join(ROOT_DIR, "level_1_ml", "train.py")], check=True)
        
    ml_model = joblib.load(ml_model_path)
    tfidf = joblib.load(tfidf_path)
    
    # Extract features
    X_test_tfidf = tfidf.transform(test_texts).toarray()
    X_test_lex = extract_lexicon_features(test_texts)
    X_test_ml = np.hstack((X_test_tfidf, X_test_lex))
    
    # Predict ML
    ml_preds = ml_model.predict(X_test_ml)
    ml_probs = ml_model.predict_proba(X_test_ml)
    
    ml_acc = accuracy_score(y_test, ml_preds)
    ml_prec, ml_rec, ml_f1, _ = precision_recall_fscore_support(y_test, ml_preds, average='macro')
    ml_roc_auc = roc_auc_score(y_test, ml_probs, multi_class='ovr')
    
    print(f"ML Baseline Accuracy: {ml_acc:.4f} | F1-Score: {ml_f1:.4f} | ROC-AUC: {ml_roc_auc:.4f}")
    
    # ------------------ EVALUATE LEVEL 2 DL MODEL ------------------
    print("\n--- Evaluating Level 2 DL LSTM Model ---")
    dl_model_path = os.path.join(CURRENT_DIR, 'models', 'lstm_model.pth')
    vocab_path = os.path.join(CURRENT_DIR, 'models', 'vocabulary.json')
    
    if not os.path.exists(dl_model_path) or not os.path.exists(vocab_path):
        print("DL LSTM weights missing. Running training now...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(CURRENT_DIR, "train_dl.py")], check=True)
        
    vocab = Vocabulary.load(vocab_path)
    X_test_dl = vocab.transform(test_texts, max_len=20)
    
    # Define model architecture and load weights
    dl_model = SentimentLSTM(
        vocab_size=vocab.num_words,
        embedding_dim=64,
        hidden_dim=64,
        output_dim=3,
        n_layers=1,
        bidirectional=True
    )
    dl_model.load_state_dict(torch.load(dl_model_path))
    dl_model.eval()
    
    # Predict DL
    with torch.no_grad():
        inputs = torch.tensor(X_test_dl, dtype=torch.long)
        logits = dl_model(inputs)
        probs = F.softmax(logits, dim=1).numpy()
        dl_preds = np.argmax(probs, axis=1)
        
    dl_acc = accuracy_score(y_test, dl_preds)
    dl_prec, dl_rec, dl_f1, _ = precision_recall_fscore_support(y_test, dl_preds, average='macro')
    dl_roc_auc = roc_auc_score(y_test, probs, multi_class='ovr')
    
    print(f"DL Model Accuracy:    {dl_acc:.4f} | F1-Score: {dl_f1:.4f} | ROC-AUC: {dl_roc_auc:.4f}")
    
    # ------------------ BENCHMARK REPORTING ------------------
    # Comparison table
    comparison_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'ML Baseline': [ml_acc, ml_prec, ml_rec, ml_f1, ml_roc_auc],
        'DL LSTM': [dl_acc, dl_prec, dl_rec, dl_f1, dl_roc_auc]
    }
    comparison_df = pd.DataFrame(comparison_data)
    # Add difference column
    comparison_df['Delta (DL - ML)'] = comparison_df['DL LSTM'] - comparison_df['ML Baseline']
    
    print("\n" + "="*40 + " COMPARISON REPORT " + "="*40)
    print(comparison_df.to_string(index=False))
    print("="*99)
    
    # Generate bar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    ml_vals = [ml_acc, ml_prec, ml_rec, ml_f1, ml_roc_auc]
    dl_vals = [dl_acc, dl_prec, dl_rec, dl_f1, dl_roc_auc]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, ml_vals, width, label='ML Baseline', color='#ff7f0e')
    ax.bar(x + width/2, dl_vals, width, label='DL LSTM', color='#1f77b4')
    
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison: ML Baseline vs. DL LSTM Model')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    chart_path = os.path.join(ROOT_DIR, 'ml_vs_dl_comparison.png')
    plt.savefig(chart_path)
    print(f"\nComparison bar chart saved as '{chart_path}'")
    
    # Save comparison table as CSV/markdown report
    report_path = os.path.join(CURRENT_DIR, 'dl_performance_report.md')
    comparison_df.to_markdown(report_path, index=False)
    print(f"Detailed report saved as '{report_path}'")

if __name__ == '__main__':
    main()
