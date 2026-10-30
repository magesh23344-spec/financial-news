# Contributing to Financial News Sentiment & Market Intelligence System

Thank you for your interest in contributing! This project is an enterprise-grade financial intelligence architecture built around a 7-Level AI Maturity Model.

## Development Workflow

1. **Fork the Repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/financial-sentiment-analysis.git
   cd financial-sentiment-analysis
   ```
3. **Set up virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS / Linux
   pip install -r requirements.txt
   ```
4. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
5. **Run the Test Suite** before committing:
   ```bash
   python3 test_suite.py
   python3 run_all.py
   ```
6. **Commit with clean conventional commit messages**:
   ```bash
   git commit -m "feat(level-2): optimize Bi-LSTM dropout layer"
   ```
7. **Push to your fork** and submit a **Pull Request**.

## Coding Standards

- Follow PEP 8 style guidelines.
- Use explicit type hints for API request and response models.
- Maintain deterministic random seeds where applicable (`torch.manual_seed`, `np.random.seed`).
- Preserve low-latency guarantees ($< 300\text{ ms}$ p95).
