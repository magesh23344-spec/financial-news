import os
import re
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize FastAPI
app = FastAPI(
    title="Financial News Sentiment API",
    description="Low-latency API to classify financial news sentiment.",
    version="1.0"
)

# Text Preprocessor (identical to training)
class TextPreprocessor:
    def __init__(self):
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
            
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        tokens = word_tokenize(text)
        cleaned_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(cleaned_tokens)

# Lexicon feature extractor (identical to training)
def extract_lexicon_features(texts):
    pos_words = {'growth', 'profit', 'upgrade', 'bullish', 'rise', 'exceed', 'partner', 'gain', 'strong', 'record'}
    neg_words = {'loss', 'decline', 'drop', 'downgrade', 'bearish', 'fall', 'probe', 'layoff', 'weak', 'pressure'}
    
    features = []
    for text in texts:
        tokens = text.lower().split()
        pos_count = sum(1 for t in tokens if t in pos_words)
        neg_count = sum(1 for t in tokens if t in neg_words)
        features.append([pos_count, neg_count, pos_count - neg_count])
    return np.array(features)

# Load pipeline files
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_sentiment_model.joblib")
VECTORIZER_PATH = os.path.join(ROOT_DIR, "models", "tfidf_vectorizer.joblib")

model = None
tfidf = None
preprocessor = TextPreprocessor()

def load_model_components():
    global model, tfidf
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "Model files not found. Please run 'python train.py' first to train and save the model."
        )
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VECTORIZER_PATH)

# Request schema
class NewsItem(BaseModel):
    headline: str

# Response schema
class PredictionResponse(BaseModel):
    headline: str
    sentiment: str
    confidence: float
    probabilities: dict

@app.on_event("startup")
def startup_event():
    try:
        load_model_components()
        print("Model components loaded successfully.")
    except Exception as e:
        print(f"Startup error: {e}")

@app.get("/health")
def health():
    if model is None or tfidf is None:
        return {"status": "Model not loaded. Call train.py."}
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(item: NewsItem):
    if model is None or tfidf is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is not initialized. Make sure you run train.py to train models before starting."
        )
    
    # Preprocess headline
    cleaned = preprocessor.preprocess(item.headline)
    
    # Vectorize text
    tfidf_feat = tfidf.transform([cleaned]).toarray()
    lex_feat = extract_lexicon_features([cleaned])
    
    # Combine features
    X = np.hstack((tfidf_feat, lex_feat))
    
    # Perform prediction
    pred_class = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    sentiment = sentiment_map[pred_class]
    confidence = float(probabilities[pred_class])
    
    prob_dict = {
        'negative': float(probabilities[0]),
        'neutral': float(probabilities[1]),
        'positive': float(probabilities[2])
    }
    
    return PredictionResponse(
        headline=item.headline,
        sentiment=sentiment,
        confidence=confidence,
        probabilities=prob_dict
    )
