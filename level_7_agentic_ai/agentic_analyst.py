import os
import json
import joblib
import numpy as np
from datetime import datetime

import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from level_1_ml.train import TextPreprocessor, extract_lexicon_features

class AgenticAnalyst:
    def __init__(self, risk_threshold=0.7):
        print("Initializing Agentic AI Financial Analyst...")
        self.risk_threshold = risk_threshold
        self.preprocessor = TextPreprocessor()
        
        # Load core scoring model from Level 1
        model_path = os.path.join(ROOT_DIR, 'models', 'best_sentiment_model.joblib')
        tfidf_path = os.path.join(ROOT_DIR, 'models', 'tfidf_vectorizer.joblib')
        if not os.path.exists(model_path) or not os.path.exists(tfidf_path):
            raise FileNotFoundError(f"Core ML components not found at '{model_path}'. Please run train.py first.")
            
        self.model = joblib.load(model_path)
        self.tfidf = joblib.load(tfidf_path)
        
    def perceive_environment(self):
        # Simulated live news feed (Perception step)
        live_feed = [
            {"id": 101, "ticker": "TSLA", "headline": "Tesla stock spikes 8% as CEO Elon Musk confirms release of new Autopilot version.", "price_volatility": 0.85},
            {"id": 102, "ticker": "AMZN", "headline": "Amazon faces legal challenges in Europe regarding cloud data privacy compliance.", "price_volatility": 0.40},
            {"id": 103, "ticker": "MSFT", "headline": "Microsoft holds normal operations meeting to vote on new company committees.", "price_volatility": 0.15},
            {"id": 104, "ticker": "AAPL", "headline": "Apple margins contract sharply due to rising chip costs, stock drops in premarket.", "price_volatility": 0.90}
        ]
        return live_feed
        
    def reason_and_plan(self, news_item):
        headline = news_item['headline']
        ticker = news_item['ticker']
        volatility = news_item['price_volatility']
        
        # 1. Run core sentiment analysis
        cleaned = self.preprocessor.preprocess(headline)
        tfidf_feat = self.tfidf.transform([cleaned]).toarray()
        lex_feat = extract_lexicon_features([cleaned])
        X = np.hstack((tfidf_feat, lex_feat))
        
        pred_class = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]
        
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        sentiment = sentiment_map[pred_class]
        confidence = probs[pred_class]
        
        # 2. Formulate decision logic (Reasoning)
        recommendation = "HOLD"
        risk_score = float(volatility * (1 - confidence)) # Combined metric: high volatility with lower confidence is higher risk
        
        # Policy rule-base for trading recommendations
        if sentiment == 'positive' and confidence >= 0.65:
            recommendation = "BUY"
        elif sentiment == 'negative' and confidence >= 0.65:
            recommendation = "SELL"
            
        # 3. Check Safety Bounds & Human-in-the-Loop constraints (Planning)
        # If volatility or risk score is higher than our safety threshold, flag for approval
        requires_approval = False
        if recommendation in ["BUY", "SELL"] and volatility > self.risk_threshold:
            requires_approval = True
            
        decision_plan = {
            "news_id": news_item['id'],
            "ticker": ticker,
            "headline": headline,
            "sentiment": sentiment,
            "confidence": float(confidence),
            "recommendation": recommendation,
            "risk_score": risk_score,
            "requires_human_approval": requires_approval,
            "status": "PENDING_APPROVAL" if requires_approval else "EXECUTED_AUTONOMOUSLY"
        }
        
        return decision_plan
        
    def execute_actions(self, plans):
        # Action step: execution logs and audit trails
        execution_records = []
        
        print("\n" + "="*45 + " AGENT ACTION EXECUTION LOG " + "="*45)
        for plan in plans:
            ticker = plan['ticker']
            rec = plan['recommendation']
            status = plan['status']
            
            print(f"\n[EVENT] News processed for {ticker}: '{plan['headline'][:50]}...'")
            print(f"  Sentiment:      {plan['sentiment'].upper()} (Confidence: {plan['confidence']:.2%})")
            print(f"  Recommendation: {rec} (Risk Score: {plan['risk_score']:.4f})")
            print(f"  Initial Status: {status}")
            
            # Simulate human-in-the-loop approval gate
            if plan['requires_human_approval']:
                print(f"  [GATE] ACTION REQUIRED: Recommendation {rec} on {ticker} exceeds volatility safety bounds! (Volatility: >70%)")
                # Auto-simulate user typing 'y' or approving
                print("  [GATE] Simulate Human Input: 'APPROVE'")
                plan['status'] = "EXECUTED_WITH_HUMAN_APPROVAL"
                print(f"  Final Status:   {plan['status']}")
            else:
                print(f"  [AUTO] Executing action autonomously. No approval gate required.")
                
            execution_records.append(plan)
            
        print("\n" + "="*118)
        return execution_records

def main():
    agent = AgenticAnalyst(risk_threshold=0.7)
    
    # 1. Perception
    feed = agent.perceive_environment()
    print(f"Perceived {len(feed)} live articles in current market cycle.")
    
    # 2. Reasoning & Planning
    plans = []
    for item in feed:
        plan = agent.reason_and_plan(item)
        plans.append(plan)
        
    # 3. Action Execution & Safety Guarding
    executed_records = agent.execute_actions(plans)
    
    # Save decision audit logs
    logs_dir = os.path.join(CURRENT_DIR, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, 'agent_decision_log.json')
    with open(log_file, 'w') as f:
        json.dump({
            "run_timestamp": datetime.now().isoformat(),
            "policy_risk_threshold": agent.risk_threshold,
            "decision_records": executed_records
        }, f, indent=4)
        
    print(f"\nSaved decision audit logs to '{log_file}'.")

if __name__ == '__main__':
    main()
