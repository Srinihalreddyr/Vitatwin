# 🧠 VitaTwin — Mental Health Intelligence Assistant

> **AI-powered early mental health detection — LLM (Ollama) + FAISS RAG + Clinical Rule Engine. Fully local, zero cloud dependency.**

VitaTwin is a clinical intelligence prototype that combines a **local LLM (Ollama/llama3)**, a **FAISS RAG pipeline**, a **multi-signal clinical rule engine**, and a **real-time Streamlit dashboard** to shift mental healthcare from reactive treatment to **early crisis detection**.

---

## 🚀 Quick Start

### Step 0 — Navigate into the project folder

```bash
cd vitatwin
```

> ⚠️ All commands below must be run from **inside** the `vitatwin/` folder, not from the parent directory.

---

### Step 1 — Install Ollama (for LLM support)

Download and install from **https://ollama.com/download**, then run:

```bash
ollama pull llama3
```

> This downloads the llama3 model (~4GB). Keep Ollama running in the background before launching the dashboard.

---

### Step 2 — Install Python dependencies

```bash
pip install faiss-cpu numpy pandas flask streamlit plotly ollama
```

> **If `faiss-cpu` fails on Windows**, try: `pip install faiss-cpu --no-cache-dir`  
> **Python 3.9 or higher** is required.

---

### Step 3 — Generate dataset + build FAISS index

```bash
python main.py setup
```

> Only needed once. Builds `data/faiss.index` and `data/faiss_meta.pkl`.

---

### Step 4a — Launch the dashboard

```bash
python -m streamlit run ui/dashboard.py
```

> Use `python -m streamlit` instead of `streamlit` directly — this avoids PATH issues on Windows.  
> Opens at **http://localhost:8501**

---

### Step 4b — OR start the REST API

```bash
python main.py api
```

> Runs at **http://localhost:5000**

---

### Step 5 — Run system tests

```bash
python main.py test
```

---

> **Without Ollama:** the system automatically falls back to a template engine — all features still work, the LLM badge in the sidebar will show 🟡 yellow instead of 🟢 green.

---

## 🏗️ Architecture

```
vitatwin/
├── data/
│   ├── generate_dataset.py   # Synthetic dataset generator (50 users, 14 days)
│   ├── users.json            # 50 user profiles
│   ├── users.csv             # Flat CSV export
│   ├── vitatwin.db           # SQLite database
│   ├── faiss.index           # FAISS vector index
│   └── faiss_meta.pkl        # TF-IDF vectorizer + document store
│
├── rag/
│   └── rag_pipeline.py       # FAISS + TF-IDF RAG engine
│
├── models/
│   ├── clinical_engine.py    # Rule-based early detection engine
│   └── assistant.py          # Ollama LLM assistant (with template fallback)
│
├── api/
│   └── server.py             # Flask REST API (6 endpoints)
│
├── ui/
│   └── dashboard.py          # Streamlit dashboard (4 pages)
│
└── main.py                   # Entry point (setup / api / test)
```

### System Flow

```
User Query
    │
    ▼
[RAG Pipeline]
TF-IDF Vectorizer → FAISS Index → Top-K User Profiles retrieved
    │
    ▼
[Clinical Rule Engine]
Multi-signal Rule Evaluation → Findings + Signals + Confidence
    │
    ▼
[Ollama LLM — llama3]
System prompt + structured patient context → Natural language response
    │
    ▼
[API / Dashboard]
JSON response with risk_score, explanation, suggested_actions, explainability block
```

---

## 📦 Part 1 — Dataset

**50 synthetic user profiles** with 14 days of longitudinal data each.

| Field | Description |
|-------|-------------|
| `user_id` | Unique identifier (VT001–VT050) |
| `mood_history` | Daily mood scores (1–10) with realistic trends |
| `stress_scores` | Daily stress scores (1–10) |
| `sleep_hours` | Daily sleep duration |
| `energy_levels` | Daily energy scores (1–10) |
| `journal_entries` | Daily written journal entries |
| `social_indicators` | Social interactions, exercise, work hours, screen time |
| `aggregates` | 14-day averages for all metrics |

**Condition labels** (distribution across 50 users):
- `healthy` (8), `mild_stress` (8), `moderate_stress` (8)
- `severe_stress` (6), `burnout_risk` (7), `anxiety_trend` (6)
- `depression_indicators` (4), `resilient` (3)

Stored in: `data/users.json`, `data/users.csv`, `data/vitatwin.db`

---

## 🔍 Part 2 — RAG Pipeline

**File:** `rag/rag_pipeline.py`

```
User Profiles
    │
    ▼
Document Builder → Clinically-engineered text chunks
    │
    ▼
TF-IDF Vectorizer (512 features, IDF-weighted, L2-normalized)
    │
    ▼
FAISS IndexFlatIP (inner-product = cosine similarity on unit vectors)
    │
    ▼
Top-K profiles → injected as context into Ollama LLM prompt
```

**Supported queries:**
- `"What mental health risks are visible?"` → retrieves high-risk profiles
- `"Has stress increased recently?"` → retrieves users with rising stress trends
- `"Summarize this user's emotional state"` → returns full longitudinal context

---

## 🤖 Part 3 — LLM Mental Health Assistant (Ollama)

**File:** `models/assistant.py`

The assistant uses **Ollama** (local LLM — llama3 by default) with FAISS-retrieved context:

```python
# RAG retrieves patient context
context = _build_user_context(user, result)   # 14-day data + clinical findings

# Ollama generates the clinical response
response = ollama.chat(
    model="llama3",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # clinical analyst persona
        {"role": "user",   "content": f"{context}\n\nQuestion: {question}"}
    ]
)
```

**Capabilities:**
- Adaptive check-in questions generated per user's risk profile by the LLM
- Supportive clinical narratives grounded in actual 14-day data
- Cross-topic reasoning (e.g. "how does stress affect sleep?")
- Graceful fallback to template engine if Ollama is not running

**LLM status** is shown live in the dashboard sidebar and AI Assistant page header.

---

## ⚠️ Part 4 — Early Detection Logic

**File:** `models/clinical_engine.py`

Four rule evaluators run in parallel:

### Burnout Detector
| Signal | Threshold | Weight |
|--------|-----------|--------|
| Avg stress (14d) | ≥ 7.0 | +15–25 pts |
| Stress trend slope | > 0.15/day | +15 pts |
| Sleep decline (7d) | ≤ −20% | +20 pts |
| Avg energy | ≤ 4.0 | +10–20 pts |
| Work hours | ≥ 10h/day | +10 pts |
| Negative journal ratio | ≥ 50% | +10–20 pts |

### Anxiety Detector
Signals: elevated stress, high stress volatility (σ), poor sleep, negative journals, social withdrawal

### Depression Detector
Signals: low mood (avg ≤ 4.0), declining mood trend, no exercise, isolation (≤ 1 social/week), very low energy, persistent negative cognition

### Resilience Detector
Identifies users with consistently strong wellness markers.

### Risk Levels
| Score | Level |
|-------|-------|
| 0–30 | LOW |
| 31–55 | MODERATE |
| 56–75 | HIGH |
| 76–100 | CRITICAL |

---

## 💡 Part 5 — Explainability

Every finding carries full provenance:

```python
Finding(
    category="burnout_risk",
    label="Potential Burnout Risk",
    risk_score=75.0,
    confidence=0.8,          # 80% — 4 of 5 signals fired
    signals=[
        Signal("avg_stress",        value=9.6,  threshold=7.0,  severity="severe"),
        Signal("low_energy",        value=3.7,  threshold=4.0,  severity="moderate"),
        Signal("overwork",          value=13.0, threshold=10.0, severity="mild"),
        Signal("negative_journals", value=0.71, threshold=0.5,  severity="severe"),
    ],
    triggered_data={
        "avg_stress_14d": 9.6,
        "stress_trend_slope": 0.039,
        "sleep_pct_change_7d": -18.2,
        "avg_energy_14d": 3.7,
        "work_hours_daily": 13.0,
        "negative_journal_ratio": 0.71,
    }
)
```

---

## 🌐 Part 6 — Clinical Intelligence API

**File:** `api/server.py` · Start: `python main.py api` → `http://localhost:5000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/mental/health` | Service health check |
| GET | `/mental/users` | List all 50 users |
| POST | `/mental/screen` | Full mental health screening |
| POST | `/mental/ask` | LLM-powered clinical Q&A |
| POST | `/mental/risk-score` | Risk score + full explainability block |
| GET | `/mental/summary/<id>` | Emotional summary + adaptive questions |
| GET | `/mental/search?q=...` | Semantic FAISS user search |

---

## 📊 Part 7 — Dashboard

**File:** `ui/dashboard.py` · Start: `streamlit run ui/dashboard.py`

| Page | Contents |
|------|----------|
| **Population Overview** | KPI cards, risk pie, condition bar chart, stress vs mood scatter, high-risk table |
| **Individual Profile** | 14-day trend charts, wellness radar, clinical findings + signals, journal entries, social context |
| **AI Assistant** | Ollama-powered chat, quick questions, adaptive check-in questions |
| **Semantic Search** | FAISS search across all 50 profiles with similarity scores |

---

## 🔧 Technology Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama (llama3 / mistral / gemma2 — auto-detected) |
| Vector DB | FAISS IndexFlatIP |
| Embeddings | Custom TF-IDF (512-dim, L2-normalized) |
| Vector Memory | FAISS-backed session memory (TF-IDF + cosine similarity) |
| Clinical Engine | Rule-based multi-signal detector |
| API | Flask |
| Dashboard | Streamlit + Plotly |
| Storage | JSON + CSV + SQLite |

---

## ⭐ Bonus — Vector Memory

**File:** `models/vector_memory.py`

VitaTwin implements **FAISS-backed vector memory** — every conversation turn (question + answer) is embedded using TF-IDF and stored in a FAISS index. On each new question, the memory is searched for semantically similar past exchanges and the most relevant ones are injected into the LLM prompt as additional context.

This gives the assistant genuine session recall — if you ask about stress and then ask a follow-up question, the assistant remembers the previous exchange and can reference it naturally.

```
New question
    │
    ▼
VectorMemory.retrieve() — FAISS search over past turns
    │
    ▼
Top-K relevant past turns → injected into LLM context block
    │
    ▼
Ollama LLM — answers with awareness of conversation history
    │
    ▼
VectorMemory.store() — new turn added to memory index
```

**Live status** is shown in the dashboard sidebar and AI Assistant header:
```
🧠 Vector Memory · 4 turns · vocab 128
```

---

## 📋 Requirements

```
faiss-cpu>=1.7.4
numpy>=1.24.0
pandas>=2.0.0
flask>=3.0.0
streamlit>=1.32.0
plotly>=5.18.0
ollama>=0.2.0
```
