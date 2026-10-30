# Low-Level Design: Automated Financial News Analysis and Market Sentiment Prediction System

This document specifies the architecture, components, data models, and API interfaces designed to support a multi-layered financial intelligence system spanning traditional ML scoring, PyTorch Deep Learning pipelines, NLP entity extraction, SLM RAG agents, and autonomous Agentic AI loops.

---

## 1. System Architecture

The system is constructed of independent microservices coordinating through RESTful APIs and asynchronous event brokers to ensure low latency for model scoring and high reliability for background workloads.

```mermaid
graph TD
    subgraph Client Application Layer
        API_GW[API Gateway / Presentation Gateway]
    end

    subgraph Synchronous Low-Latency Services
        API_GW -->|POST /predict| CORE_ML[ML/DL Model Serving Microservice]
        CORE_ML -->|Query/Cache| REDIS[(In-Memory Redis Cache)]
    end

    subgraph Asynchronous Context & SLM Services
        API_GW -->|POST /qna| SLM_SVC[SLM Q&A Microservice]
        API_GW -->|POST /summarize| SLM_SVC
        SLM_SVC -->|Retrieve Context| VECT_DB[(Context Store / Vector Database)]
    end

    subgraph Background Processing Workers
        FEEDS[Financial News Feeds / APIs] -->|Ingest News| INGEST[Ingestion Microservice]
        INGEST -->|Write Raw| DB[(System Relational Database)]
        INGEST -->|Event Message| BROKER[[RabbitMQ / Kafka Broker]]
        
        BROKER -->|Process Headline| PREP_WKR[Preprocessing & ML Scoring Worker]
        PREP_WKR -->|Update Sentiment| DB
        
        BROKER -->|Process Document| NLP_WKR[NLP Document Parser Worker]
        NLP_WKR -->|Extract Entities & Summaries| VECT_DB
        NLP_WKR -->|Update Context| DB
    end

    subgraph Autonomous Analysis
        AGENT[Agentic AI Analyst Orchestrator] -->|Poll Feeds & Scores| DB
        AGENT -->|Query Semantic Context| VECT_DB
        AGENT -->|Generate Summaries / Q&A| SLM_SVC
        AGENT -->|Publish Recommendations| DB
        AGENT -->|Request Approval| API_GW
    end

    style CORE_ML fill:#d1e7dd,stroke:#0f5132,stroke-width:2px
    style SLM_SVC fill:#cfe2ff,stroke:#084298,stroke-width:2px
    style AGENT fill:#f8d7da,stroke:#842029,stroke-width:2px
```

---

## 2. API Contract Specifications (OpenAPI v3)

### 2.1 Synchronous Sentiment Prediction Endpoint
* **Endpoint:** `POST /v1/predict`
* **Content-Type:** `application/json`
* **Request Payload Schema:**
```json
{
  "title": "API Request payload",
  "type": "object",
  "properties": {
    "headline": {
      "type": "string",
      "description": "Raw text of the headline to analyze."
    },
    "model_version": {
      "type": "string",
      "default": "dl_lstm_v1",
      "description": "Selected model engine: ml_baseline_v1 or dl_lstm_v1."
    }
  },
  "required": ["headline"]
}
```
* **Response Payload Schema (200 OK):**
```json
{
  "headline": "Apple shares surge 5% following blockbuster earnings release",
  "sentiment": "positive",
  "confidence": 0.8921,
  "probabilities": {
    "negative": 0.0245,
    "neutral": 0.0834,
    "positive": 0.8921
  },
  "model_engine": "dl_lstm_v1",
  "processed_at": "2026-08-07T11:27:00Z"
}
```

### 2.2 Asynchronous Document Summarization Endpoint
* **Endpoint:** `POST /v1/summarize`
* **Content-Type:** `application/json`
* **Request Payload Schema:**
```json
{
  "document_text": "Full text of the earnings call transcript or analyst report...",
  "ticker_reference": "MSFT"
}
```
* **Response Payload Schema (202 Accepted):**
```json
{
  "task_id": "task_abc123xyz",
  "status": "processing",
  "estimated_duration_sec": 3.5
}
```

---

## 3. Relational Database & Context Store Schema

```mermaid
erDiagram
    NEWS_ARTICLES ||--|| SENTIMENT_SCORES : "has score"
    NEWS_ARTICLES ||--o{ EXTRACTED_ENTITIES : "contains"
    NEWS_ARTICLES }o--|| COMPANIES : "references"
    RECOMMENDATIONS }o--|| SENTIMENT_SCORES : "generated from"
    RECOMMENDATIONS }o--|| COMPANIES : "targets"

    NEWS_ARTICLES {
        bigint id PK
        varchar headline
        text body_text
        timestamp published_at
        varchar source
        boolean is_synthetic
    }

    SENTIMENT_SCORES {
        bigint id PK
        bigint article_id FK
        varchar sentiment_label
        double confidence
        double prob_positive
        double prob_negative
        double prob_neutral
        varchar model_engine
    }

    EXTRACTED_ENTITIES {
        bigint id PK
        bigint article_id FK
        varchar entity_type
        varchar entity_value
        double confidence_score
    }

    COMPANIES {
        bigint id PK
        varchar ticker UNIQUE
        varchar name
        varchar sector
    }

    RECOMMENDATIONS {
        bigint id PK
        bigint company_id FK
        varchar recommendation_type
        text rationale
        double risk_score
        boolean human_approved
        varchar status
        timestamp created_at
    }
```

---

## 4. Sequence Diagrams

### 4.1 Real-Time News Ingestion & Scoring Flow
```mermaid
sequenceDiagram
    autonumber
    participant Feeds as Ingestion Connector
    participant Queue as Event Broker
    participant Worker as Scoring Worker
    participant DB as Relational DB
    participant Cache as Redis Cache

    Feeds->>DB: Save Raw News Metadata
    Feeds->>Queue: Publish Ingested News Event
    Queue->>Worker: Consume News Event
    Worker->>Worker: Clean Text & Tokenize
    Worker->>Worker: Run Model Inference (ML/DL)
    Worker->>DB: Save Sentiment Score & Probabilities
    Worker->>Cache: Cache Sentiment for Fast UI Lookup
```

### 4.2 Autonomous Agentic Decision Sequence
```mermaid
sequenceDiagram
    autonumber
    participant Agent as Agent Analyst Orchestrator
    participant DB as Relational DB
    participant VDB as Context/Vector DB
    participant SLM as SLM Summarizer
    participant Gate as Approval Gate API

    Agent->>DB: Query fresh Sentiment Scores
    Agent->>VDB: Retrieve semantic reports & entities
    Agent->>SLM: Request summary of related earnings transcripts
    SLM-->>Agent: Returns Summary & Grounded Context
    Agent->>Agent: Reason on Sentiment, Risk metrics, & Context
    Agent->>Agent: Generate Plan (e.g. Issue Alert / Buy / Sell)
    Agent->>DB: Save planned recommendation (Status: PENDING_APPROVAL)
    Agent->>Gate: Trigger Approval Alert to Gateway
    Gate-->>Agent: Human User approves action
    Agent->>DB: Update recommendation (Status: APPROVED)
```
