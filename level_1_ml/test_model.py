import os
import sys
import joblib
import numpy as np

# Ensure level_1_ml is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from train import TextPreprocessor, extract_lexicon_features

def test_inference():
    print("Initializing test inference script...")
    
    # Load model and vectorizer
    model_path = os.path.join(ROOT_DIR, 'models', 'best_sentiment_model.joblib')
    vectorizer_path = os.path.join(ROOT_DIR, 'models', 'tfidf_vectorizer.joblib')
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print(f"Error: Trained model files not found in '{model_path}'. Please run 'python train.py' first.")
        return
        
    model = joblib.load(model_path)
    tfidf = joblib.load(vectorizer_path)
    preprocessor = TextPreprocessor()
    
    test_headlines = [
        "Google shares skyrocket following record-breaking cloud revenues.",
        "Tesla faces federal probe over autopilot system malfunctions, stock falls.",
        "Amazon is planning to release its standard corporate update tomorrow.",
        "Federal reserve signals interest rates will remain unchanged for the next quarter.",
        "Apple announces layoffs of 500 contract workers in global restructuring."
    ]
    
    print("\nPredicting sentiment for test headlines:")
    print("-" * 80)
    
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    for headline in test_headlines:
        cleaned = preprocessor.preprocess(headline)
        
        tfidf_feat = tfidf.transform([cleaned]).toarray()
        lex_feat = extract_lexicon_features([cleaned])
        X = np.hstack((tfidf_feat, lex_feat))
        
        pred_class = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        
        sentiment = sentiment_map[pred_class]
        confidence = probs[pred_class]
        
        print(f"Headline:   {headline}")
        print(f"Prediction: {sentiment} (Confidence: {confidence:.2%})")
        print(f"Probs:      Negative: {probs[0]:.2%}, Neutral: {probs[1]:.2%}, Positive: {probs[2]:.2%}")
        print("-" * 80)

if __name__ == '__main__':
    test_inference()
