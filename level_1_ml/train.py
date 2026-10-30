import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Download NLTK resources if not already present
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# 1. GENERATE SYNTHETIC DATA
# To allow immediate running in VS Code, we generate realistic financial news data.
def generate_synthetic_data(num_samples=1000):
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'BRK.A', 'JPM']
    
    pos_templates = [
        "shares rise after {ticker} reports record-breaking Q3 earnings, exceeding revenue guidance.",
        "{ticker} upgraded to Buy by top Wall Street analysts, citing strong product demand.",
        "{ticker} announces strategic partnership to expand artificial intelligence capabilities.",
        "investors bullish on {ticker} as profit margins increase by 15% year-over-year.",
        "breakthrough technology launch positions {ticker} for long-term sector dominance."
    ]
    
    neg_templates = [
        "{ticker} stock drops following weak quarterly revenue forecast and supply chain bottleneck.",
        "{ticker} downgraded to Sell due to rising operational expenses and margin pressure.",
        "regulatory probe initiated against {ticker} over antitrust and data compliance concerns.",
        "shares of {ticker} fall as sales growth decelerates in key international markets.",
        "{ticker} CEO announces layoffs and restructuring plans amid macroeconomic headwinds."
    ]
    
    neu_templates = [
        "{ticker} to hold annual shareholder meeting next week to vote on board members.",
        "{ticker} announces routine dividend payment of $0.25 per share.",
        "{ticker} files standard Form 4 with SEC detailing executive stock transactions.",
        "{ticker} launches minor software update resolving general performance bugs.",
        "analysts maintain neutral rating on {ticker} ahead of upcoming product cycle."
    ]
    
    data = []
    np.random.seed(42)
    
    for i in range(num_samples):
        ticker = np.random.choice(tickers)
        label_class = np.random.choice([0, 1, 2], p=[0.25, 0.50, 0.25]) # 0: Negative, 1: Neutral, 2: Positive
        
        if label_class == 2:
            template = np.random.choice(pos_templates)
            sentiment_label = 'positive'
            # Simulating stock price return (positive mean)
            price_return = np.random.normal(0.025, 0.015)
        elif label_class == 0:
            template = np.random.choice(neg_templates)
            sentiment_label = 'negative'
            # Simulating stock price return (negative mean)
            price_return = np.random.normal(-0.025, 0.015)
        else:
            template = np.random.choice(neu_templates)
            sentiment_label = 'neutral'
            # Simulating stock price return (neutral mean)
            price_return = np.random.normal(0.001, 0.005)
            
        headline = template.format(ticker=ticker)
        
        # Add random noise/chatter to text to make it realistic
        noise = " " + np.random.choice([
            "Market watches closely.", "Stock remains active in pre-market trading.", 
            "Volume remains average.", "No comment from IR team.", ""
        ])
        headline += noise
        
        # Mock timestamp within the last 30 days
        timestamp = pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 30), hours=np.random.randint(0, 24))
        
        data.append({
            'timestamp': timestamp,
            'ticker': ticker,
            'headline': headline,
            'sentiment': sentiment_label,
            'price_return_24h': price_return
        })
        
    df = pd.DataFrame(data)
    # Sort chronologically to prevent data leakage during time splits
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# 2. TEXT PREPROCESSING PIPELINE
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess(self, text):
        # Remove special characters and digits, keep letters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Lowercase
        text = text.lower()
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stop words and lemmatize
        cleaned_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(cleaned_tokens)

# 3. FEATURE ENGINEERING WITH WEAK FINANCIAL LEXICON
def extract_lexicon_features(texts):
    # Simulated Loughran-McDonald mini lexicon
    pos_words = {'growth', 'profit', 'upgrade', 'bullish', 'rise', 'exceed', 'partner', 'gain', 'strong', 'record'}
    neg_words = {'loss', 'decline', 'drop', 'downgrade', 'bearish', 'fall', 'probe', 'layoff', 'weak', 'pressure'}
    
    features = []
    for text in texts:
        tokens = text.lower().split()
        pos_count = sum(1 for t in tokens if t in pos_words)
        neg_count = sum(1 for t in tokens if t in neg_words)
        
        features.append([
            pos_count,
            neg_count,
            pos_count - neg_count # Net sentiment score
        ])
    return np.array(features)

# 4. TRAINING & EVALUATION PIPELINE
def main():
    print("Generating synthetic financial news dataset...")
    df = generate_synthetic_data(1200)
    print(f"Dataset generated with {len(df)} samples.")
    print("Class Distribution:\n", df['sentiment'].value_counts())
    
    # Preprocess text
    print("\nPreprocessing news text...")
    preprocessor = TextPreprocessor()
    df['cleaned_headline'] = df['headline'].apply(preprocessor.preprocess)
    
    # Map sentiment to integers
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    y = df['sentiment'].map(sentiment_map).values
    
    # 80/20 train/test split based on time index (sequential split is crucial to prevent leakage)
    split_idx = int(len(df) * 0.8)
    
    train_texts = df['cleaned_headline'].iloc[:split_idx]
    test_texts = df['cleaned_headline'].iloc[split_idx:]
    
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    print(f"\nSplitting data: {len(train_texts)} train, {len(test_texts)} test (Chronological split)")
    
    # Extract TF-IDF Features
    print("Vectorizing text using TF-IDF...")
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(train_texts).toarray()
    X_test_tfidf = tfidf.transform(test_texts).toarray()
    
    # Extract Lexicon Features
    print("Extracting domain lexicon features...")
    X_train_lex = extract_lexicon_features(train_texts)
    X_test_lex = extract_lexicon_features(test_texts)
    
    # Combine Features
    X_train = np.hstack((X_train_tfidf, X_train_lex))
    X_test = np.hstack((X_test_tfidf, X_test_lex))
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Support Vector Machine': SVC(probability=True, class_weight='balanced', random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_f1 = -1
    best_model_name = ""
    best_model_obj = None
    
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        
        # Calculate ROC-AUC (One-vs-Rest)
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        
        results[name] = {
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        }
        
        print(f"\n--- {name} ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model

    # Save results as a table
    results_df = pd.DataFrame(results).T
    print("\nPerformance Comparison Table:")
    print(results_df.to_string())
    
    # Save best model components
    models_dir = os.path.join(ROOT_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_model_obj, os.path.join(models_dir, 'best_sentiment_model.joblib'))
    joblib.dump(tfidf, os.path.join(models_dir, 'tfidf_vectorizer.joblib'))
    print(f"\nSaved Best Model ({best_model_name}) and TF-IDF Vectorizer to '{models_dir}'.")
    
    # Let's generate a confusion matrix plot for the best model
    best_preds = best_model_obj.predict(X_test)
    cm = confusion_matrix(y_test, best_preds)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {best_model_name}')
    plt.colorbar()
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, ['Negative', 'Neutral', 'Positive'])
    plt.yticks(tick_marks, ['Negative', 'Neutral', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Fill cells with numbers
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
                 
    plt.tight_layout()
    cm_path = os.path.join(ROOT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion Matrix plot saved as '{cm_path}'")
    
    # Show classification report for the best model
    print(f"\nClassification Report for {best_model_name}:")
    print(classification_report(y_test, best_preds, target_names=['Negative', 'Neutral', 'Positive']))

if __name__ == '__main__':
    main()
