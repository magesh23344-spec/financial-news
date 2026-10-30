import os
import sys
import time
import json
import unittest
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

class TestFinancialIntelligencePipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("\n" + "="*80)
        print(" INITIALIZING END-TO-END VALIDATION TEST SUITE (7 LEVELS)")
        print("="*80)

    def test_level_1_ml_baseline(self):
        print("\n[TEST L1] Testing Core ML Baseline...")
        from level_1_ml.train import TextPreprocessor, extract_lexicon_features
        import joblib
        import numpy as np

        model_path = os.path.join(ROOT_DIR, 'models', 'best_sentiment_model.joblib')
        vectorizer_path = os.path.join(ROOT_DIR, 'models', 'tfidf_vectorizer.joblib')
        self.assertTrue(os.path.exists(model_path), "Level 1 Model file missing")
        self.assertTrue(os.path.exists(vectorizer_path), "TF-IDF Vectorizer missing")

        model = joblib.load(model_path)
        tfidf = joblib.load(vectorizer_path)
        preprocessor = TextPreprocessor()

        test_headline = "Apple reports record revenue and profit margins."
        cleaned = preprocessor.preprocess(test_headline)
        X = np.hstack((tfidf.transform([cleaned]).toarray(), extract_lexicon_features([cleaned])))
        
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        self.assertEqual(pred, 2, "Expected Positive Sentiment (class 2)")
        self.assertGreater(probs[2], 0.70, "Confidence should exceed 70%")
        print("  ✓ Level 1 ML Baseline passed (Positive detected with high confidence)")

    def test_level_2_dl_lstm(self):
        print("\n[TEST L2] Testing PyTorch Bi-LSTM Deep Learning...")
        import torch
        import torch.nn.functional as F
        from level_1_ml.train import TextPreprocessor
        from level_2_dl.train_dl import Vocabulary, SentimentLSTM

        weights_path = os.path.join(ROOT_DIR, 'level_2_dl', 'models', 'lstm_model.pth')
        vocab_path = os.path.join(ROOT_DIR, 'level_2_dl', 'models', 'vocabulary.json')
        self.assertTrue(os.path.exists(weights_path), "LSTM weights missing")
        self.assertTrue(os.path.exists(vocab_path), "Vocabulary missing")

        vocab = Vocabulary.load(vocab_path)
        model = SentimentLSTM(
            vocab_size=vocab.num_words,
            embedding_dim=64,
            hidden_dim=64,
            output_dim=3,
            n_layers=1,
            bidirectional=True
        )
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        model.eval()

        preprocessor = TextPreprocessor()
        test_headline = "Tesla stock drops following weak quarterly revenue forecast and supply chain bottleneck."
        cleaned = preprocessor.preprocess(test_headline)
        seq = vocab.transform([cleaned], max_len=20)
        
        with torch.no_grad():
            logits = model(torch.tensor(seq, dtype=torch.long))
            probs = F.softmax(logits, dim=1).numpy()[0]
            pred = int(probs.argmax())

        self.assertEqual(pred, 0, "Expected Negative Sentiment (class 0)")
        print("  ✓ Level 2 Bi-LSTM passed (Negative detected correctly)")

    def test_level_3_nlp_pipeline(self):
        print("\n[TEST L3] Testing Contextual NLP & Entity Extraction...")
        from level_3_nlp.nlp_pipeline import FinancialNLPParser
        parser = FinancialNLPParser()

        sample_text = "Satya Nadella said Microsoft Corp. (MSFT) Azure revenue grew by 18 percent to $25.3 billion."
        entities = parser.extract_entities(sample_text)

        self.assertIn('MSFT', entities['TICKERS'], "Ticker MSFT not extracted")
        self.assertIn('Microsoft', entities['COMPANIES'], "Company Microsoft not extracted")
        self.assertIn('Satya Nadella', entities['EXECUTIVES'], "Executive Satya Nadella not extracted")
        self.assertTrue(len(entities['METRICS']) >= 1, "Monetary/Percentage metrics not extracted")
        print("  ✓ Level 3 NLP Entity Extraction passed (Tickers, Companies, Execs, Metrics extracted)")

    def test_level_4_slm_rag(self):
        print("\n[TEST L4] Testing Small Language Model (SLM) & RAG...")
        from level_4_slm.slm_inference import SmallLanguageModelAgent, retrieve_relevant_context

        mock_context = [
            {"text": "Azure cloud services grew by 18 percent.", "sentiment": "positive"},
            {"text": "Xbox hardware sales fell by 6 percent.", "sentiment": "negative"}
        ]
        context = retrieve_relevant_context("What happened with Azure cloud services?", mock_context)
        self.assertIn("18 percent", context)

        agent = SmallLanguageModelAgent()
        answer = agent.answer_question("How much did Azure cloud revenue grow?", context)
        self.assertTrue(len(answer) > 20, "Answer was empty or too short")
        print("  ✓ Level 4 SLM RAG Query passed")

    def test_level_5_lld_document(self):
        print("\n[TEST L5] Validating Architecture & Low-Level Design Specification...")
        lld_path = os.path.join(ROOT_DIR, 'level_5_lld', 'architecture_lld.md')
        self.assertTrue(os.path.exists(lld_path), "LLD file missing")
        with open(lld_path, 'r') as f:
            content = f.read()
        self.assertIn("System Architecture", content)
        self.assertIn("POST /v1/predict", content)
        print("  ✓ Level 5 LLD Architecture document verified")

    def test_level_6_stress_testing(self):
        print("\n[TEST L6] Testing Generative AI Stress Cases...")
        from level_6_gen_ai.synthetic_generator import generate_synthetic_stress_cases
        cases = generate_synthetic_stress_cases()
        self.assertEqual(len(cases), 5, "Expected 5 adversarial stress cases")
        print(f"  ✓ Level 6 GenAI Stress Suite validated ({len(cases)} edge-cases configured)")

    def test_level_7_agentic_ai_safety_gate(self):
        print("\n[TEST L7] Testing Autonomous Agent Loop & Safety Gate...")
        from level_7_agentic_ai.agentic_analyst import AgenticAnalyst
        analyst = AgenticAnalyst(risk_threshold=0.7)

        # High-volatility news triggering human gate
        high_risk_news = {
            "id": 999,
            "ticker": "HIGH_VOL",
            "headline": "Apple margins contract sharply due to rising chip costs, stock drops in premarket.",
            "price_volatility": 0.95
        }
        plan = analyst.reason_and_plan(high_risk_news)
        self.assertTrue(plan['requires_human_approval'], "Safety Gate should trigger for high volatility (> 0.70)")
        self.assertEqual(plan['status'], "PENDING_APPROVAL")
        print("  ✓ Level 7 Agentic AI Safety Gate passed (High-volatility action flagged for approval)")

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFinancialIntelligencePipeline)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    
    if result.wasSuccessful():
        print("\n" + "*"*80)
        print(" ALL 7 LEVEL TESTS PASSED WITH 100% SUCCESS!")
        print(" Ready for GitHub push and production deployment.")
        print("*"*80 + "\n")
        sys.exit(0)
    else:
        print("\n[FAIL] Some tests did not pass.")
        sys.exit(1)
