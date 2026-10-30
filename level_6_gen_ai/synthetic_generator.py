import os
import json
import numpy as np
import joblib
import torch
import torch.nn.functional as F

import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from level_1_ml.train import TextPreprocessor, extract_lexicon_features
from level_2_dl.train_dl import Vocabulary, SentimentLSTM

def generate_synthetic_stress_cases():
    # Subtle, complex, and adversarial financial news cases
    stress_cases = [
        {
            "headline": "Company reports record revenues but increasing operational costs drag margins down by 8 percent.",
            "expected_sentiment": "negative",
            "notes": "Mixed sentiment: positive revenue, negative margins"
        },
        {
            "headline": "CEO says layoffs will not prevent future drops in growth.",
            "expected_sentiment": "negative",
            "notes": "Negation phrase: 'will not prevent future drops'"
        },
        {
            "headline": "Antitrust lawsuit dismissed, positioning company for immediate market recovery.",
            "expected_sentiment": "positive",
            "notes": "Negation of a negative event: 'lawsuit dismissed'"
        },
        {
            "headline": "Stock price surges to record highs despite sector-wide macro depression.",
            "expected_sentiment": "positive",
            "notes": "Adversarial contrast: positive stock, negative sector"
        },
        {
            "headline": "Apple announces standard executive meeting to detail minor software upgrades.",
            "expected_sentiment": "neutral",
            "notes": "Routine news with positive-adjacent keywords like 'upgrades'"
        }
    ]
    return stress_cases

def main():
    print("Initializing Generative AI Stress-Testing Module...")
    stress_cases = generate_synthetic_stress_cases()
    print(f"Generated {len(stress_cases)} synthetic stress-test scenarios.")
    
    # ------------------ LOAD MODElS ------------------
    # Load ML
    ml_model_path = os.path.join(ROOT_DIR, 'models', 'best_sentiment_model.joblib')
    tfidf_path = os.path.join(ROOT_DIR, 'models', 'tfidf_vectorizer.joblib')
    if not os.path.exists(ml_model_path) or not os.path.exists(tfidf_path):
        print("ML Baseline model not found. Run train.py first.")
        return
    ml_model = joblib.load(ml_model_path)
    tfidf = joblib.load(tfidf_path)
    
    # Load DL
    dl_model_path = os.path.join(ROOT_DIR, 'level_2_dl', 'models', 'lstm_model.pth')
    vocab_path = os.path.join(ROOT_DIR, 'level_2_dl', 'models', 'vocabulary.json')
    if not os.path.exists(dl_model_path) or not os.path.exists(vocab_path):
        print("DL LSTM model not found. Run train_dl.py first.")
        return
    vocab = Vocabulary.load(vocab_path)
    dl_model = SentimentLSTM(
        vocab_size=vocab.num_words,
        embedding_dim=64,
        hidden_dim=64,
        output_dim=3,
        n_layers=1,
        bidirectional=True
    )
    dl_model.load_state_dict(torch.load('level_2_dl/models/lstm_model.pth'))
    dl_model.eval()
    
    preprocessor = TextPreprocessor()
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    print("\n" + "="*50 + " MODEL STRESS TEST RESULTS " + "="*50)
    print(f"{'Headline':<65} | {'Expected':<10} | {'ML Predict':<10} | {'DL Predict':<10} | {'Pass/Fail':<10}")
    print("-" * 115)
    
    report = []
    passed_ml = 0
    passed_dl = 0
    
    for case in stress_cases:
        headline = case['headline']
        expected = case['expected_sentiment']
        
        # Preprocess
        cleaned = preprocessor.preprocess(headline)
        
        # 1. ML Prediction
        tfidf_feat = tfidf.transform([cleaned]).toarray()
        lex_feat = extract_lexicon_features([cleaned])
        X_ml = np.hstack((tfidf_feat, lex_feat))
        ml_pred_class = ml_model.predict(X_ml)[0]
        ml_pred = sentiment_map[ml_pred_class]
        
        # 2. DL Prediction
        X_dl = vocab.transform([cleaned], max_len=20)
        with torch.no_grad():
            inputs = torch.tensor(X_dl, dtype=torch.long)
            logits = dl_model(inputs)
            probs = F.softmax(logits, dim=1).numpy()[0]
            dl_pred_class = np.argmax(probs)
            dl_pred = sentiment_map[dl_pred_class]
            
        ml_pass = "PASS" if ml_pred == expected else "FAIL"
        dl_pass = "PASS" if dl_pred == expected else "FAIL"
        
        if ml_pass == "PASS": passed_ml += 1
        if dl_pass == "PASS": passed_dl += 1
        
        # Display cropped headline if too long
        disp_headline = headline[:62] + "..." if len(headline) > 65 else headline
        print(f"{disp_headline:<65} | {expected:<10} | {ml_pred:<10} | {dl_pred:<10} | ML: {ml_pass} / DL: {dl_pass}")
        
        report.append({
            'headline': headline,
            'expected_sentiment': expected,
            'notes': case['notes'],
            'ml_prediction': ml_pred,
            'ml_pass': ml_pass == "PASS",
            'dl_prediction': dl_pred,
            'dl_pass': dl_pass == "PASS"
        })
        
    print("=" * 115)
    print(f"ML Baseline Stress Test Pass Rate: {passed_ml}/{len(stress_cases)} ({passed_ml/len(stress_cases):.1%})")
    print(f"DL LSTM Model Stress Test Pass Rate: {passed_dl}/{len(stress_cases)} ({passed_dl/len(stress_cases):.1%})")
    
    # Save stress test report
    outputs_dir = os.path.join(CURRENT_DIR, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    report_file = os.path.join(outputs_dir, 'stress_test_report.json')
    with open(report_file, 'w') as f:
        json.dump({
            'ml_pass_rate': passed_ml / len(stress_cases),
            'dl_pass_rate': passed_dl / len(stress_cases),
            'cases': report
        }, f, indent=4)
        
    print(f"\nSaved detailed stress-testing report to '{report_file}'.")

if __name__ == '__main__':
    main()
