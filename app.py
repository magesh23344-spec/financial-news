import os
import sys
import time
import json
import torch
import torch.nn.functional as F
import joblib
import numpy as np
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Setup root paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import domain modules
from level_1_ml.train import TextPreprocessor, extract_lexicon_features
from level_2_dl.train_dl import Vocabulary, SentimentLSTM
from level_3_nlp.nlp_pipeline import FinancialNLPParser
from level_4_slm.slm_inference import SmallLanguageModelAgent, retrieve_relevant_context, load_context_store
from level_6_gen_ai.synthetic_generator import generate_synthetic_stress_cases
from level_7_agentic_ai.agentic_analyst import AgenticAnalyst

# Initialize FastAPI App
app = FastAPI(
    title="Financial News Sentiment & Market Intelligence System",
    description="7-Level Automated Financial Intelligence Architecture (ML Baseline -> PyTorch DL -> NLP -> SLM RAG -> LLD -> GenAI Stress-Testing -> Agentic AI)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State & Lazy Loaded Models
preprocessor = TextPreprocessor()
nlp_parser = FinancialNLPParser()
slm_agent = SmallLanguageModelAgent()

ml_model = None
tfidf_vectorizer = None
dl_model = None
dl_vocab = None

def init_models():
    global ml_model, tfidf_vectorizer, dl_model, dl_vocab
    
    # 1. Load ML Model
    ml_path = os.path.join(ROOT_DIR, 'models', 'best_sentiment_model.joblib')
    tfidf_path = os.path.join(ROOT_DIR, 'models', 'tfidf_vectorizer.joblib')
    if os.path.exists(ml_path) and os.path.exists(tfidf_path):
        ml_model = joblib.load(ml_path)
        tfidf_vectorizer = joblib.load(tfidf_path)
        print("ML Baseline components loaded successfully.")
    else:
        print("ML Model components not found on disk.")
        
    # 2. Load DL Model
    dl_weights = os.path.join(ROOT_DIR, 'level_2_dl', 'models', 'lstm_model.pth')
    vocab_path = os.path.join(ROOT_DIR, 'level_2_dl', 'models', 'vocabulary.json')
    if os.path.exists(dl_weights) and os.path.exists(vocab_path):
        dl_vocab = Vocabulary.load(vocab_path)
        dl_model = SentimentLSTM(
            vocab_size=dl_vocab.num_words,
            embedding_dim=64,
            hidden_dim=64,
            output_dim=3,
            n_layers=1,
            bidirectional=True
        )
        dl_model.load_state_dict(torch.load(dl_weights, map_location=torch.device('cpu')))
        dl_model.eval()
        print("Deep Learning LSTM components loaded successfully.")
    else:
        print("DL LSTM components not found on disk.")

@app.on_event("startup")
def startup():
    init_models()

# Pydantic Schemas
class SentimentRequest(BaseModel):
    headline: str

class TranscriptRequest(BaseModel):
    transcript: str

class RAGQueryRequest(BaseModel):
    query: str
    custom_context: Optional[str] = None

class AgentFeedItem(BaseModel):
    id: int
    ticker: str
    headline: str
    price_volatility: float

class AgentApprovalRequest(BaseModel):
    news_id: int
    decision: str # "APPROVE" or "REJECT"

# ----------------- API ENDPOINTS -----------------

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "system": "7-Level Financial Intelligence Architecture",
        "ml_loaded": ml_model is not None,
        "dl_loaded": dl_model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/predict/ml")
def predict_ml(req: SentimentRequest):
    if ml_model is None or tfidf_vectorizer is None:
        raise HTTPException(status_code=503, detail="ML Model not loaded. Run level_1_ml/train.py first.")
        
    start_t = time.perf_counter()
    cleaned = preprocessor.preprocess(req.headline)
    tfidf_feat = tfidf_vectorizer.transform([cleaned]).toarray()
    lex_feat = extract_lexicon_features([cleaned])
    X = np.hstack((tfidf_feat, lex_feat))
    
    pred_idx = int(ml_model.predict(X)[0])
    probs = ml_model.predict_proba(X)[0]
    latency_ms = (time.perf_counter() - start_t) * 1000
    
    labels = ["negative", "neutral", "positive"]
    return {
        "engine": "Level 1 - Core ML Baseline (Logistic Regression / Ensemble)",
        "headline": req.headline,
        "sentiment": labels[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {
            "negative": round(float(probs[0]), 4),
            "neutral": round(float(probs[1]), 4),
            "positive": round(float(probs[2]), 4)
        },
        "latency_ms": round(latency_ms, 2)
    }

@app.post("/api/predict/dl")
def predict_dl(req: SentimentRequest):
    if dl_model is None or dl_vocab is None:
        raise HTTPException(status_code=503, detail="DL LSTM Model not loaded. Run level_2_dl/train_dl.py first.")
        
    start_t = time.perf_counter()
    cleaned = preprocessor.preprocess(req.headline)
    seq = dl_vocab.transform([cleaned], max_len=20)
    
    with torch.no_grad():
        inputs = torch.tensor(seq, dtype=torch.long)
        logits = dl_model(inputs)
        probs = F.softmax(logits, dim=1).numpy()[0]
        pred_idx = int(np.argmax(probs))
        
    latency_ms = (time.perf_counter() - start_t) * 1000
    labels = ["negative", "neutral", "positive"]
    return {
        "engine": "Level 2 - Deep Learning PyTorch Bidirectional LSTM",
        "headline": req.headline,
        "sentiment": labels[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {
            "negative": round(float(probs[0]), 4),
            "neutral": round(float(probs[1]), 4),
            "positive": round(float(probs[2]), 4)
        },
        "latency_ms": round(latency_ms, 2)
    }

@app.post("/api/predict/benchmark")
def benchmark_prediction(req: SentimentRequest):
    ml_res = predict_ml(req)
    dl_res = predict_dl(req)
    
    agreement = ml_res["sentiment"] == dl_res["sentiment"]
    return {
        "headline": req.headline,
        "agreement": agreement,
        "ml": ml_res,
        "dl": dl_res,
        "consensus_recommendation": ml_res["sentiment"] if agreement else f"DIVERGENT (ML: {ml_res['sentiment']}, DL: {dl_res['sentiment']})"
    }

@app.post("/api/nlp/extract")
def extract_nlp(req: TranscriptRequest):
    segments = nlp_parser.segment_transcript(req.transcript)
    parsed_segments = []
    
    pos_words = {'strong', 'grow', 'exceed', 'pay off', 'beat', 'growth', 'profit', 'rise', 'record'}
    neg_words = {'warn', 'constraint', 'negatively', 'fall', 'pressure', 'loss', 'drop', 'layoff', 'weak'}
    
    for idx, seg in enumerate(segments):
        entities = nlp_parser.extract_entities(seg)
        words = seg.lower().split()
        p_score = sum(1 for w in words if w in pos_words)
        n_score = sum(1 for w in words if w in neg_words)
        
        sent = "positive" if p_score > n_score else ("negative" if n_score > p_score else "neutral")
        parsed_segments.append({
            "segment_id": idx + 1,
            "text": seg,
            "entities": entities,
            "sentiment": sent
        })
        
    return {
        "total_segments": len(parsed_segments),
        "segments": parsed_segments
    }

@app.post("/api/slm/query")
def slm_rag_query(req: RAGQueryRequest):
    if req.custom_context:
        context = req.custom_context
    else:
        context_store = load_context_store()
        context = retrieve_relevant_context(req.query, context_store)
        
    answer = slm_agent.answer_question(req.query, context)
    return {
        "query": req.query,
        "grounded_context": context,
        "answer": answer
    }

@app.post("/api/slm/summarize")
def slm_summarize(req: TranscriptRequest):
    summary = slm_agent.generate_summary(req.transcript)
    return {
        "original_length": len(req.transcript.split()),
        "summary": summary
    }

@app.get("/api/stress-test")
def run_stress_test():
    stress_cases = generate_synthetic_stress_cases()
    results = []
    
    passed_ml = 0
    passed_dl = 0
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    for case in stress_cases:
        h = case['headline']
        expected = case['expected_sentiment']
        
        # ML
        cleaned = preprocessor.preprocess(h)
        tfidf_feat = tfidf_vectorizer.transform([cleaned]).toarray()
        lex_feat = extract_lexicon_features([cleaned])
        X = np.hstack((tfidf_feat, lex_feat))
        ml_p = sentiment_map[int(ml_model.predict(X)[0])]
        ml_pass = ml_p == expected
        if ml_pass: passed_ml += 1
        
        # DL
        seq = dl_vocab.transform([cleaned], max_len=20)
        with torch.no_grad():
            logits = dl_model(torch.tensor(seq, dtype=torch.long))
            probs = F.softmax(logits, dim=1).numpy()[0]
            dl_p = sentiment_map[int(np.argmax(probs))]
        dl_pass = dl_p == expected
        if dl_pass: passed_dl += 1
        
        results.append({
            "headline": h,
            "notes": case['notes'],
            "expected": expected,
            "ml_pred": ml_p,
            "ml_pass": ml_pass,
            "dl_pred": dl_p,
            "dl_pass": dl_pass
        })
        
    return {
        "total_cases": len(stress_cases),
        "ml_pass_rate": round(passed_ml / len(stress_cases), 2),
        "dl_pass_rate": round(passed_dl / len(stress_cases), 2),
        "cases": results
    }

@app.get("/api/agent/cycle")
def run_agentic_cycle():
    analyst = AgenticAnalyst(risk_threshold=0.7)
    feed = analyst.perceive_environment()
    plans = []
    for item in feed:
        plan = analyst.reason_and_plan(item)
        plans.append(plan)
    executed = analyst.execute_actions(plans)
    return {
        "status": "success",
        "market_cycle_time": datetime.now().isoformat(),
        "policy_risk_threshold": 0.7,
        "items_analyzed": len(executed),
        "records": executed
    }

# ----------------- PRESENTATION WEB DASHBOARD -----------------

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    # Modern Glassmorphic Dashboard with 7 Interactive Station Tabs
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial News Sentiment & Market Intelligence System</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #111827;
            --bg-card: rgba(17, 24, 39, 0.75);
            --border-card: rgba(255, 255, 255, 0.08);
            --accent-cyan: #00f2fe;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
            --glass-blur: blur(16px);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-main);
            min-height: 100vh;
            line-height: 1.6;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 15% 20%, rgba(59, 130, 246, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 85% 80%, rgba(139, 92, 246, 0.12) 0%, transparent 40%);
        }

        header {
            border-bottom: 1px solid var(--border-card);
            background: rgba(10, 14, 23, 0.85);
            backdrop-filter: var(--glass-blur);
            position: sticky;
            top: 0;
            z-index: 100;
            padding: 1rem 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }

        .brand-badge {
            background: linear-gradient(135deg, #00f2fe, #4facfe);
            color: #000;
            font-weight: 800;
            font-size: 0.85rem;
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            letter-spacing: 0.5px;
        }

        .brand-title {
            font-size: 1.25rem;
            font-weight: 700;
            background: linear-gradient(90deg, #fff, #cbd5e1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header-links {
            display: flex;
            gap: 1rem;
            align-items: center;
        }

        .btn-link {
            color: var(--text-muted);
            text-decoration: none;
            font-size: 0.9rem;
            font-weight: 500;
            transition: color 0.2s;
        }

        .btn-link:hover {
            color: var(--accent-cyan);
        }

        .btn-pill {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-card);
            padding: 0.4rem 1rem;
            border-radius: 20px;
            color: #fff;
            text-decoration: none;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            transition: all 0.2s;
        }

        .btn-pill:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }

        /* Hero Banner */
        .hero {
            padding: 3rem 2rem 2rem;
            max-width: 1300px;
            margin: 0 auto;
            text-align: center;
        }

        .hero h1 {
            font-size: 2.8rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin-bottom: 0.8rem;
            background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero p {
            color: var(--text-muted);
            font-size: 1.15rem;
            max-width: 800px;
            margin: 0 auto 1.5rem;
        }

        .level-pills {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 0.5rem;
            margin-bottom: 2.5rem;
        }

        .pill {
            font-size: 0.78rem;
            padding: 0.3rem 0.8rem;
            border-radius: 9999px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-card);
            color: #cbd5e1;
            font-family: 'JetBrains Mono', monospace;
        }

        .pill.active-pill {
            border-color: var(--accent-cyan);
            background: rgba(0, 242, 254, 0.08);
            color: var(--accent-cyan);
        }

        /* Main Container & Tabs */
        .container {
            max-width: 1300px;
            margin: 0 auto;
            padding: 0 2rem 4rem;
        }

        .nav-tabs {
            display: flex;
            gap: 0.5rem;
            border-bottom: 1px solid var(--border-card);
            margin-bottom: 2rem;
            overflow-x: auto;
            padding-bottom: 0.5rem;
        }

        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 0.75rem 1.25rem;
            font-size: 0.95rem;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            white-space: nowrap;
        }

        .tab-btn:hover {
            color: #fff;
            background: rgba(255, 255, 255, 0.04);
        }

        .tab-btn.active {
            color: #000;
            background: var(--accent-cyan);
            box-shadow: 0 0 20px rgba(0, 242, 254, 0.4);
        }

        .tab-pane {
            display: none;
            animation: fadeIn 0.3s ease-in-out;
        }

        .tab-pane.active {
            display: block;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Glass Cards */
        .card {
            background: var(--bg-card);
            backdrop-filter: var(--glass-blur);
            border: 1px solid var(--border-card);
            border-radius: 14px;
            padding: 1.8rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.2rem;
            border-bottom: 1px solid var(--border-card);
            padding-bottom: 0.8rem;
        }

        .card-title {
            font-size: 1.25rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        .grid-3 {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.2rem;
        }

        @media(max-width: 900px) {
            .grid-2, .grid-3 {
                grid-template-columns: 1fr;
            }
        }

        /* Input Controls */
        textarea, input[type="text"], select {
            width: 100%;
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid var(--border-card);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            color: #fff;
            font-family: inherit;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
        }

        textarea:focus, input[type="text"]:focus {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 10px rgba(0, 242, 254, 0.2);
        }

        .btn-action {
            background: linear-gradient(135deg, #00f2fe, #4facfe);
            color: #000;
            border: none;
            padding: 0.85rem 1.6rem;
            border-radius: 8px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .btn-action:hover {
            opacity: 0.9;
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(0, 242, 254, 0.4);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.08);
            color: #fff;
            border: 1px solid var(--border-card);
            padding: 0.6rem 1rem;
            border-radius: 6px;
            font-size: 0.85rem;
            cursor: pointer;
            transition: background 0.2s;
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.15);
        }

        .sample-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.8rem 0 1.2rem;
        }

        .chip {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-card);
            padding: 0.35rem 0.75rem;
            border-radius: 6px;
            font-size: 0.8rem;
            cursor: pointer;
            color: var(--text-muted);
            transition: all 0.2s;
        }

        .chip:hover {
            border-color: var(--accent-cyan);
            color: #fff;
            background: rgba(0, 242, 254, 0.05);
        }

        /* Result Displays */
        .result-box {
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid var(--border-card);
            border-radius: 10px;
            padding: 1.25rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            min-height: 180px;
        }

        .badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .badge-positive { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }
        .badge-negative { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }
        .badge-neutral { background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid #f59e0b; }

        .progress-bar-container {
            margin-top: 0.6rem;
        }

        .progress-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            margin-bottom: 0.2rem;
        }

        .progress-track {
            height: 8px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            transition: width 0.5s ease-out;
        }

        .progress-fill.positive { background: var(--accent-green); }
        .progress-fill.negative { background: var(--accent-red); }
        .progress-fill.neutral { background: var(--accent-amber); }

        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }

        th, td {
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border-card);
        }

        th {
            background: rgba(255, 255, 255, 0.03);
            color: var(--text-muted);
            font-weight: 600;
        }

        tr:hover {
            background: rgba(255, 255, 255, 0.02);
        }

        /* Agent Station Log */
        .agent-log {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            background: #050811;
            padding: 1.2rem;
            border-radius: 8px;
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid var(--border-card);
        }

        .log-entry {
            margin-bottom: 0.8rem;
            padding-bottom: 0.8rem;
            border-bottom: 1px dashed rgba(255, 255, 255, 0.06);
        }

        /* Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(8px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }

        .modal-card {
            background: #111827;
            border: 1px solid var(--accent-amber);
            border-radius: 12px;
            max-width: 500px;
            width: 90%;
            padding: 2rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
        }

        footer {
            border-top: 1px solid var(--border-card);
            padding: 2rem;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
        }
    </style>
</head>
<body>

    <header>
        <div class="brand">
            <span class="brand-badge">FIN-AI 2.0</span>
            <span class="brand-title">Financial News Sentiment & Market Intelligence System</span>
        </div>
        <div class="header-links">
            <a href="/docs" target="_blank" class="btn-pill">⚡ Swagger API</a>
            <a href="https://github.com" target="_blank" class="btn-pill">📦 GitHub Repo</a>
        </div>
    </header>

    <section class="hero">
        <h1>7 Levels of Market Intelligence</h1>
        <p>A production-ready financial NLP & autonomous trading intelligence architecture spanning classical ML baselines, PyTorch deep learning, domain RAG, adversarial stress testing, and autonomous agent loops.</p>
        <div class="level-pills">
            <span class="pill active-pill">L1: ML Baseline (TF-IDF)</span>
            <span class="pill active-pill">L2: Deep Learning (Bi-LSTM)</span>
            <span class="pill active-pill">L3: Contextual NLP Parser</span>
            <span class="pill active-pill">L4: Small Language Model RAG</span>
            <span class="pill active-pill">L5: Low-Level System Design</span>
            <span class="pill active-pill">L6: GenAI Stress-Testing</span>
            <span class="pill active-pill">L7: Autonomous Agentic Loop</span>
        </div>
    </section>

    <div class="container">
        <!-- Navigation Tabs -->
        <div class="nav-tabs">
            <button class="tab-btn active" onclick="switchTab('tab-predict')">⚔️ ML vs DL Arena</button>
            <button class="tab-btn" onclick="switchTab('tab-nlp')">🔍 NLP Transcript Parser</button>
            <button class="tab-btn" onclick="switchTab('tab-slm')">🧠 SLM & RAG Q&A</button>
            <button class="tab-btn" onclick="switchTab('tab-stress')">🎯 GenAI Stress Lab</button>
            <button class="tab-btn" onclick="switchTab('tab-agent')">🤖 Autonomous Trading Agent</button>
            <button class="tab-btn" onclick="switchTab('tab-system')">📐 System Architecture</button>
        </div>

        <!-- TAB 1: PREDICTION ARENA -->
        <div id="tab-predict" class="tab-pane active">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Real-Time Financial Headline Inference Arena</div>
                    <span style="font-size:0.85rem; color:var(--text-muted)">Evaluates TF-IDF ML vs. PyTorch Bi-LSTM</span>
                </div>
                
                <div class="sample-chips">
                    <span class="chip" onclick="setHeadline('Apple shares surge 5% following blockbuster earnings release and record revenue.')">🚀 Earnings Beat (AAPL)</span>
                    <span class="chip" onclick="setHeadline('Tesla faces federal regulatory probe over autopilot safety, stock drops sharply.')">⚠️ Regulatory Probe (TSLA)</span>
                    <span class="chip" onclick="setHeadline('Microsoft schedules annual shareholder voting meeting to review committee.')">📅 Routine Neutral (MSFT)</span>
                    <span class="chip" onclick="setHeadline('Company reports record revenues but increasing operational costs drag margins down by 8 percent.')">⚡ Mixed Adversarial News</span>
                </div>

                <textarea id="headlineInput" rows="2" placeholder="Type or paste any financial news headline here...">Apple shares surge 5% following blockbuster earnings release and record revenue.</textarea>
                
                <div style="margin-top: 1rem; display: flex; gap: 1rem; align-items: center;">
                    <button class="btn-action" onclick="runBenchmark()">⚡ Run Real-Time Benchmark</button>
                    <span id="arenaStatus" style="font-size: 0.85rem; color: var(--text-muted);">Ready. p95 target < 300ms.</span>
                </div>
            </div>

            <div class="grid-2">
                <!-- ML Baseline Card -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Level 1: Core ML Baseline</div>
                        <span id="mlLatency" class="pill">Latency: -- ms</span>
                    </div>
                    <div id="mlResult" class="result-box">
                        <p style="color:var(--text-muted);">Run inference to view Level 1 ML classification, Loughran-McDonald lexicon sentiment, and class probabilities.</p>
                    </div>
                </div>

                <!-- DL LSTM Card -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Level 2: PyTorch Deep Learning (Bi-LSTM)</div>
                        <span id="dlLatency" class="pill">Latency: -- ms</span>
                    </div>
                    <div id="dlResult" class="result-box">
                        <p style="color:var(--text-muted);">Run inference to view Level 2 Bidirectional LSTM sequence classification and dense token probabilities.</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 2: NLP PARSER -->
        <div id="tab-nlp" class="tab-pane">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Level 3: Earnings Call Transcript & Named Entity Parser</div>
                    <button class="btn-secondary" onclick="runNLP()">Run Parser</button>
                </div>
                <p style="color:var(--text-muted); margin-bottom: 1rem;">Segments raw financial filings or audio call transcripts, extracting financial metrics ($/%), recognized stock tickers, corporate entities, and executive personnel.</p>
                <textarea id="transcriptInput" rows="4">Microsoft Corp. (MSFT) announced its Q3 earnings call today. CEO Satya Nadella started the call by highlighting strong demand for Azure cloud services, stating that cloud margins grew by 18 percent this quarter. Our AI infrastructure investments are paying off, and Azure revenue exceeded $25.3 billion. However, CFO Amy Hood warned that supply chain constraints could negatively impact hardware shipments. Xbox hardware sales fell by 6 percent, resulting in minor margin pressure. Overall, Microsoft expects to beat guidance for the next fiscal year.</textarea>
                
                <div style="margin-top:1.5rem;" id="nlpOutputContainer">
                    <div id="nlpResultsTable"></div>
                </div>
            </div>
        </div>

        <!-- TAB 3: SLM & RAG -->
        <div id="tab-slm" class="tab-pane">
            <div class="grid-2">
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Level 4: Financial RAG Q&A Engine</div>
                    </div>
                    <p style="color:var(--text-muted); margin-bottom: 1rem;">Queries the transcript knowledge base with semantic grounding.</p>
                    
                    <div class="sample-chips">
                        <span class="chip" onclick="setQuery('What did the CEO say about Azure cloud revenue and margins?')">Azure Cloud Growth?</span>
                        <span class="chip" onclick="setQuery('Why did Xbox hardware sales drop?')">Xbox Sales Drop?</span>
                    </div>

                    <input type="text" id="ragQueryInput" value="What did the CEO say about Azure cloud revenue and margins?" style="margin-bottom: 1rem;">
                    <button class="btn-action" onclick="runRAG()">🔍 Ask RAG Engine</button>

                    <div style="margin-top:1.5rem;">
                        <div id="ragAnswerBox" class="result-box">
                            <p style="color:var(--text-muted);">Results will appear here...</p>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Automated Executive Summarization</div>
                    </div>
                    <p style="color:var(--text-muted); margin-bottom: 1rem;">Compresses multi-page financial filings into high-density executive bulletins.</p>
                    <button class="btn-secondary" onclick="runSummarize()" style="margin-bottom:1rem;">Generate Summary</button>
                    <div id="summaryBox" class="result-box">
                        <p style="color:var(--text-muted);">Click 'Generate Summary' to run extractive/abstractive financial ranker...</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 4: GENAI STRESS LAB -->
        <div id="tab-stress" class="tab-pane">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Level 6: Generative AI Adversarial Stress-Testing Lab</div>
                    <button class="btn-action" onclick="runStressTest()">🎯 Execute Stress Suite</button>
                </div>
                <p style="color:var(--text-muted); margin-bottom: 1.2rem;">Adversarial evaluation testing models against complex linguistic phenomena: double negations ("not prevent drops"), conflicting sentiment clauses, and macro-sector contrasts.</p>
                <div id="stressTestTableContainer">
                    <div class="result-box">Click 'Execute Stress Suite' to evaluate both ML & DL models against adversarial edge-cases.</div>
                </div>
            </div>
        </div>

        <!-- TAB 5: AGENTIC AI -->
        <div id="tab-agent" class="tab-pane">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Level 7: Autonomous Agentic AI Financial Analyst Station</div>
                    <button class="btn-action" onclick="runAgentCycle()">⚡ Run Agent Perception Loop</button>
                </div>
                <p style="color:var(--text-muted); margin-bottom: 1rem;">
                    Autonomous Perception-Reasoning-Planning-Action loop. The agent ingests live ticker streams, performs confidence scoring, assesses volatility-adjusted risk, recommends [BUY / HOLD / SELL], and invokes a <strong>Human-in-the-Loop Safety Gate</strong> when risk bounds are exceeded.
                </p>

                <div class="agent-log" id="agentConsole">
                    <p style="color:#6b7280;">Agent standby. Ready to trigger perception cycle...</p>
                </div>
            </div>
        </div>

        <!-- TAB 6: SYSTEM ARCHITECTURE -->
        <div id="tab-system" class="tab-pane">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Level 5: Low-Level Design (LLD) & Microservices Topology</div>
                </div>
                <div class="grid-3" style="margin-bottom:1.5rem;">
                    <div class="card" style="margin-bottom:0;">
                        <h4 style="color:var(--accent-cyan); margin-bottom:0.5rem;">Synchronous Tier</h4>
                        <p style="font-size:0.85rem; color:var(--text-muted);">RESTful FastAPI inference endpoints. In-memory TF-IDF and PyTorch Bi-LSTM scoring with < 300ms p95 latency.</p>
                    </div>
                    <div class="card" style="margin-bottom:0;">
                        <h4 style="color:var(--accent-purple); margin-bottom:0.5rem;">Asynchronous Workers</h4>
                        <p style="font-size:0.85rem; color:var(--text-muted);">Broker-driven background tasks (Kafka/RabbitMQ) parsing continuous wire feeds and updating the Vector/Context Store.</p>
                    </div>
                    <div class="card" style="margin-bottom:0;">
                        <h4 style="color:var(--accent-green); margin-bottom:0.5rem;">Autonomous Loop</h4>
                        <p style="font-size:0.85rem; color:var(--text-muted);">Agentic orchestrator performing continuous market perception, risk calculations, and trade signal dispatching with safety gates.</p>
                    </div>
                </div>

                <div class="result-box" style="line-height:1.8;">
                    <strong>API Endpoints Exposed:</strong><br>
                    • <code>POST /api/predict/ml</code> — Level 1 Core ML Sentiment Classifier<br>
                    • <code>POST /api/predict/dl</code> — Level 2 PyTorch Bi-LSTM Deep Learning<br>
                    • <code>POST /api/predict/benchmark</code> — Side-by-side comparative inference<br>
                    • <code>POST /api/nlp/extract</code> — Financial entity recognition & segment sentiment<br>
                    • <code>POST /api/slm/query</code> — Context-grounded RAG Q&A<br>
                    • <code>POST /api/slm/summarize</code> — Financial filing summarization<br>
                    • <code>GET  /api/stress-test</code> — Adversarial robustness validation suite<br>
                    • <code>GET  /api/agent/cycle</code> — Autonomous Agentic loop execution
                </div>
            </div>
        </div>
    </div>

    <footer>
        <p>Built with ❤️ using Python 3.11, PyTorch, Scikit-learn, NLTK, and FastAPI. Engineered for High-Frequency Decision Support.</p>
    </footer>

    <script>
        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        }

        function setHeadline(text) {
            document.getElementById('headlineInput').value = text;
        }

        function setQuery(text) {
            document.getElementById('ragQueryInput').value = text;
        }

        async function runBenchmark() {
            const headline = document.getElementById('headlineInput').value.trim();
            if (!headline) return;

            document.getElementById('arenaStatus').innerText = "Running inference...";
            try {
                const res = await fetch('/api/predict/benchmark', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({headline: headline})
                });
                const data = await res.json();
                
                // ML Result
                const ml = data.ml;
                document.getElementById('mlLatency').innerText = `Latency: ${ml.latency_ms} ms`;
                document.getElementById('mlResult').innerHTML = renderSentimentResult(ml);

                // DL Result
                const dl = data.dl;
                document.getElementById('dlLatency').innerText = `Latency: ${dl.latency_ms} ms`;
                document.getElementById('dlResult').innerHTML = renderSentimentResult(dl);

                document.getElementById('arenaStatus').innerText = `Consensus: ${data.consensus_recommendation}`;
            } catch(e) {
                document.getElementById('arenaStatus').innerText = "Error running benchmark: " + e;
            }
        }

        function renderSentimentResult(item) {
            const badgeClass = item.sentiment === 'positive' ? 'badge-positive' : (item.sentiment === 'negative' ? 'badge-negative' : 'badge-neutral');
            const posPct = (item.probabilities.positive * 100).toFixed(1);
            const neuPct = (item.probabilities.neutral * 100).toFixed(1);
            const negPct = (item.probabilities.negative * 100).toFixed(1);

            return `
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                    <div>Prediction: <span class="badge ${badgeClass}">${item.sentiment}</span></div>
                    <div>Confidence: <strong>${(item.confidence * 100).toFixed(1)}%</strong></div>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-label"><span>Positive</span><span>${posPct}%</span></div>
                    <div class="progress-track"><div class="progress-fill positive" style="width:${posPct}%"></div></div>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-label"><span>Neutral</span><span>${neuPct}%</span></div>
                    <div class="progress-track"><div class="progress-fill neutral" style="width:${neuPct}%"></div></div>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-label"><span>Negative</span><span>${negPct}%</span></div>
                    <div class="progress-track"><div class="progress-fill negative" style="width:${negPct}%"></div></div>
                </div>
            `;
        }

        async function runNLP() {
            const text = document.getElementById('transcriptInput').value.trim();
            const res = await fetch('/api/nlp/extract', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({transcript: text})
            });
            const data = await res.json();
            
            let html = `<table><thead><tr><th>#</th><th>Segment Text</th><th>Entities Detected</th><th>Sentiment</th></tr></thead><tbody>`;
            data.segments.forEach(s => {
                const badgeClass = s.sentiment === 'positive' ? 'badge-positive' : (s.sentiment === 'negative' ? 'badge-negative' : 'badge-neutral');
                let entStr = [];
                if (s.entities.TICKERS.length) entStr.push(`<strong>Tickers:</strong> ${s.entities.TICKERS.join(', ')}`);
                if (s.entities.COMPANIES.length) entStr.push(`<strong>Companies:</strong> ${s.entities.COMPANIES.join(', ')}`);
                if (s.entities.EXECUTIVES.length) entStr.push(`<strong>Execs:</strong> ${s.entities.EXECUTIVES.join(', ')}`);
                if (s.entities.METRICS.length) entStr.push(`<strong>Metrics:</strong> ${s.entities.METRICS.join(', ')}`);
                
                html += `<tr>
                    <td>${s.segment_id}</td>
                    <td style="font-size:0.85rem;">${s.text}</td>
                    <td style="font-size:0.8rem;">${entStr.length ? entStr.join('<br>') : '<span style="color:#6b7280;">None</span>'}</td>
                    <td><span class="badge ${badgeClass}">${s.sentiment}</span></td>
                </tr>`;
            });
            html += `</tbody></table>`;
            document.getElementById('nlpResultsTable').innerHTML = html;
        }

        async function runRAG() {
            const query = document.getElementById('ragQueryInput').value.trim();
            const res = await fetch('/api/slm/query', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: query})
            });
            const data = await res.json();
            document.getElementById('ragAnswerBox').innerHTML = `
                <div style="margin-bottom:0.8rem; color:var(--accent-cyan);"><strong>Grounding Context:</strong></div>
                <div style="font-size:0.85rem; color:#cbd5e1; margin-bottom:1rem; background:rgba(255,255,255,0.03); padding:0.6rem; border-radius:6px;">"${data.grounded_context}"</div>
                <div style="color:var(--accent-green); margin-bottom:0.4rem;"><strong>Answer:</strong></div>
                <div style="font-size:0.9rem; line-height:1.5;">${data.answer.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')}</div>
            `;
        }

        async function runSummarize() {
            const text = document.getElementById('transcriptInput').value.trim();
            const res = await fetch('/api/slm/summarize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({transcript: text})
            });
            const data = await res.json();
            document.getElementById('summaryBox').innerHTML = `
                <div style="color:var(--accent-cyan); margin-bottom:0.6rem;"><strong>High-Density Executive Brief:</strong></div>
                <div style="font-size:0.9rem; line-height:1.6; color:#f1f5f9;">${data.summary}</div>
            `;
        }

        async function runStressTest() {
            const container = document.getElementById('stressTestTableContainer');
            container.innerHTML = "<div class='result-box'>Running adversarial stress test scenarios against ML and DL architectures...</div>";
            
            const res = await fetch('/api/stress-test');
            const data = await res.json();
            
            let html = `
                <div style="display:flex; gap:1.5rem; margin-bottom:1rem;">
                    <div>ML Baseline Pass Rate: <strong style="color:var(--accent-cyan);">${(data.ml_pass_rate*100).toFixed(0)}%</strong></div>
                    <div>DL LSTM Pass Rate: <strong style="color:var(--accent-green);">${(data.dl_pass_rate*100).toFixed(0)}%</strong></div>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Adversarial Headline</th>
                            <th>Linguistic Trap / Notes</th>
                            <th>Expected</th>
                            <th>ML Predict</th>
                            <th>DL Predict</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            data.cases.forEach(c => {
                const mlPassTag = c.ml_pass ? '<span style="color:#10b981;">✓ PASS</span>' : '<span style="color:#ef4444;">✗ FAIL</span>';
                const dlPassTag = c.dl_pass ? '<span style="color:#10b981;">✓ PASS</span>' : '<span style="color:#ef4444;">✗ FAIL</span>';
                html += `
                    <tr>
                        <td style="font-size:0.85rem;">${c.headline}</td>
                        <td style="font-size:0.8rem; color:#94a3b8;">${c.notes}</td>
                        <td><span class="badge">${c.expected}</span></td>
                        <td>${c.ml_pred} (${mlPassTag})</td>
                        <td>${c.dl_pred} (${dlPassTag})</td>
                    </tr>
                `;
            });
            html += `</tbody></table>`;
            container.innerHTML = html;
        }

        async function runAgentCycle() {
            const consoleBox = document.getElementById('agentConsole');
            consoleBox.innerHTML = "<p style='color:#00f2fe;'>[AGENT] Initializing perception loop. Ingesting live financial feed...</p>";
            
            const res = await fetch('/api/agent/cycle');
            const data = await res.json();
            
            let logHtml = `<div style="color:#00f2fe; margin-bottom:1rem;">[MARKET CYCLE STARTED: ${data.market_cycle_time}]</div>`;
            data.records.forEach(r => {
                const recColor = r.recommendation === 'BUY' ? '#10b981' : (r.recommendation === 'SELL' ? '#ef4444' : '#f59e0b');
                const gateBadge = r.requires_human_approval 
                    ? '<span style="color:#f59e0b; background:rgba(245,158,11,0.15); padding:2px 6px; border-radius:4px;">⚠️ HUMAN APPROVAL TRIGGERED</span>' 
                    : '<span style="color:#10b981;">✓ AUTONOMOUS EXECUTION</span>';
                    
                logHtml += `
                    <div class="log-entry">
                        <div><strong>[TICKER: ${r.ticker}]</strong> ${r.headline}</div>
                        <div style="margin-top:0.3rem;">
                            Sentiment: <strong>${r.sentiment.toUpperCase()}</strong> (${(r.confidence*100).toFixed(1)}% conf) | 
                            Action: <strong style="color:${recColor};">${r.recommendation}</strong> | 
                            Risk Score: <strong>${r.risk_score.toFixed(4)}</strong>
                        </div>
                        <div style="margin-top:0.3rem;">Execution Status: ${gateBadge} (${r.status})</div>
                    </div>
                `;
            });
            logHtml += `<div style="color:#10b981; margin-top:1rem;">[AGENT] Cycle concluded. Audit ledger committed.</div>`;
            consoleBox.innerHTML = logHtml;
        }

        // Initialize default view
        window.onload = function() {
            runBenchmark();
        };
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == '__main__':
    import uvicorn
    print("\n" + "="*80)
    print(" STARTING UNIFIED FINANCIAL INTELLIGENCE API & INTERACTIVE DASHBOARD")
    print(" Web UI Available at: http://127.0.0.1:8000/ or http://127.0.0.1:8000/dashboard")
    print(" Swagger API Docs at: http://127.0.0.1:8000/docs")
    print("="*80 + "\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
