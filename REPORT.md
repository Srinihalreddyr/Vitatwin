# VitaTwin Mental Health Intelligence System
## Technical Report — Internship Evaluation Submission

---

## 1. Executive Summary

VitaTwin is a prototype AI-powered mental health intelligence system designed to shift clinical practice from reactive care to early detection. The system integrates a **local LLM (Ollama/llama3)**, a **FAISS RAG pipeline**, a **multi-signal clinical rule engine**, a **Flask REST API**, and a **real-time Streamlit dashboard** — all running fully locally with no cloud dependency or API keys.

The core insight motivating the design is that early mental health deterioration is a **pattern recognition problem over longitudinal data**. Deteriorating sleep, rising stress, increasingly negative journal entries, and social withdrawal — each individually unremarkable — together constitute a high-confidence early warning signal. VitaTwin operationalises this insight in a clinically transparent, explainable system where every output is traceable to the specific data values that triggered it.

---

## 2. Architecture Overview

The system is composed of five integrated layers:

**Layer 1 — Data Layer**
50 synthetic user profiles, each containing 14 days of longitudinal observations across mood, stress, sleep, energy, journal entries, and social/behavioural indicators. Stored in JSON (for the RAG pipeline), CSV (for analytics), and SQLite (for relational queries). Condition labels are distributed realistically across eight categories, ensuring the retrieval system returns meaningful clinical variance.

**Layer 2 — RAG Pipeline (FAISS + TF-IDF)**
Each user profile is converted into a clinically-engineered text document. A custom TF-IDF vectoriser (512 features, IDF-weighted, L2-normalised) produces dense vectors indexed with FAISS `IndexFlatIP` (inner product on unit vectors = cosine similarity). At query time, the same vectoriser encodes the query and FAISS returns the top-K most semantically similar profiles in milliseconds. The retrieved profiles are then formatted as structured context and injected into the LLM prompt — this is the genuine RAG loop: **retrieve → augment → generate**.

**Layer 3 — Clinical Rule Engine**
Four independent rule evaluators (burnout, anxiety, depression, resilience) analyse each user's time-series data against clinically-informed thresholds. Each evaluator produces a `Finding` object carrying the signals that fired, the raw values that triggered each signal, severity classifications, confidence scores, and suggested actions. This layer runs before the LLM and feeds its structured output into the LLM context — grounding the LLM's responses in verified clinical findings.

**Layer 4 — LLM Assistant (Ollama)**
The assistant uses Ollama to run llama3 (or mistral/gemma2) locally. For each question, it: (1) retrieves the user's profile via FAISS, (2) runs the clinical engine to generate findings, (3) builds a structured context block containing the full 14-day time-series, aggregates, journal entries, social indicators, and clinical engine findings, then (4) calls `ollama.chat()` with a clinical analyst system prompt and the structured context as user content. The LLM reasons over real patient data to produce natural language clinical narratives. A template-based fallback activates automatically if Ollama is not running.

**Layer 5 — API & Dashboard**
A Flask REST API exposes seven clinical endpoints. A Streamlit dashboard provides four views covering population-level analytics, individual user deep-dives, the LLM chat interface, and semantic search.

---

## 3. RAG Design

The retrieval pipeline implements the standard RAG architecture with one key design choice: **document construction is clinically engineered**.

Each user's document is composed as coherent clinical prose rather than raw JSON field concatenation. For example: *"average stress 9.6 out of ten trend increasing, sleep declining, work hours 13.0 daily, journals showing 71% negative sentiment"*. This aligns the TF-IDF vocabulary directly with clinical query language — when a clinician asks "users showing burnout", the query tokens match the document tokens precisely.

The three required RAG queries function as follows:

- **"What mental health risks are visible?"** — FAISS retrieves the highest-risk profiles; the LLM synthesises their clinical findings into a structured risk narrative.
- **"Has stress increased recently?"** — FAISS retrieves profiles with rising stress trends; the LLM reports the trend slope, week-over-week change, and clinical significance.
- **"Summarize this user's emotional state"** — FAISS retrieves the specific user's profile; the LLM generates a multi-paragraph summary covering mood, stress, sleep, energy, social engagement, and clinical assessment.

The RAG-retrieved context is always visible to the professor in the dashboard sidebar (`🟢 Ollama · llama3`) and in the AI Assistant page header, making the LLM involvement explicit.

---

## 4. LLM Approach

The LLM integration uses Ollama as the local inference engine with llama3 as the preferred model. The integration is designed around three principles:

**Grounded generation.** The LLM never generates clinical claims from parametric memory alone. Every response is conditioned on the structured patient context built from FAISS-retrieved data. The system prompt instructs the model to reference specific data points in its responses and to frame all outputs as clinical observations rather than diagnoses.

**Structured context injection.** The patient context passed to the LLM includes:
- 14-day daily time-series for all four biomarkers
- 14-day aggregate averages and trend directions
- Last three journal entries
- Social and lifestyle indicators
- Clinical engine findings with signal names and severities
- Suggested actions from the rule engine

This means the LLM is reasoning over genuinely rich longitudinal clinical data, not just a summary.

**Adaptive clinical questioning.** The `adaptive_questions()` method sends the full patient context to the LLM with a prompt asking it to generate three personalised check-in questions for this specific patient today. This produces clinically relevant, patient-specific questions rather than generic wellness prompts.

**LLM status display.** The sidebar and AI Assistant header show the live LLM status (`🟢 Ollama · llama3` or `🟡 Template fallback`) so the evaluator can immediately verify the LLM is active.

---

## 5. Explainability Logic

The explainability system follows the **signal chain-of-custody** principle: every output is traceable to specific input values that crossed specific thresholds.

For each finding the system records:

| Field | Purpose |
|-------|---------|
| `signals[]` | Each rule that fired — name, value, threshold, severity, description |
| `triggered_data` | Raw numerical values at time of evaluation |
| `confidence` | Ratio of signals fired to maximum possible (0.0–1.0) |
| `explanation` | Human-readable narrative of which signals drove the finding |
| `suggested_actions` | Clinically appropriate intervention recommendations |

This design is visible directly in the dashboard: the Clinical Findings section shows each finding as a card with a confidence bar, colour-coded signal rows (SEVERE / MODERATE / MILD), the raw trigger data as a JSON object, and the suggested actions. A reviewer can follow the complete path from raw data value → threshold crossing → signal → finding → recommended action.

The API's `/mental/risk-score` endpoint returns a full `explainability` block:

```json
{
  "explainability": {
    "signals_used": ["avg_stress", "low_energy", "overwork", "negative_journals"],
    "data_triggered": {"avg_stress_14d": 9.6, "work_hours_daily": 13.0},
    "confidence_level": "80%",
    "methodology": "VitaTwin Clinical Rule Engine v1.0"
  }
}
```

---

## 6. Clinical Validity

The rule thresholds are informed by established clinical frameworks:

- **Burnout**: sustained high stress + energy depletion + overwork — consistent with Maslach's burnout inventory model
- **Anxiety**: stress volatility (σ) + sleep disruption + social withdrawal — consistent with GAD diagnostic criteria clusters
- **Depression**: persistent low mood + declining trend + energy depletion + social isolation + negative cognition — consistent with DSM-5 indicator clusters

This system is a **screening tool**, not a diagnostic instrument. All findings use language such as "indicators detected" and "potential risk" rather than clinical diagnoses. The suggested actions consistently recommend professional follow-up for high-risk findings.

---

## 7. Product Thinking

**Multi-signal aggregation over single-metric alerts.** Stress at 7/10 alone is not alarming. Stress at 7/10 combined with 20% sleep decline, 5 consecutive negative journal entries, and social isolation — that is an early crisis signal. The product's core value is precisely this aggregation across time and domains.

**API-first design for clinical augmentation.** The `/mental/risk-score` endpoint returns not just a score but the complete explainability block. This is deliberate: the system is designed to augment clinicians, not replace them. A clinician receiving a risk alert needs the *why* — the explainability block delivers that in a structured, parseable form that can be rendered in any clinical interface.

**Local-first architecture.** Running entirely on Ollama with no cloud dependency means this system could be deployed in a clinical environment with strict data privacy requirements (HIPAA, GDPR) without any data leaving the hospital network.

**Graceful degradation.** If Ollama is not running, the system continues to function with template-based responses. The dashboard communicates this clearly. This ensures the system is always available even without the LLM, which is important in clinical contexts where reliability is non-negotiable.

---

## 8. Evaluation Criteria Coverage

| Criterion | Implementation |
|-----------|---------------|
| **LLM integration** | Ollama (llama3) called via `ollama.chat()` for all assistant responses and adaptive questions |
| **RAG implementation** | FAISS + TF-IDF retrieval; top-K profiles injected as LLM context |
| **Mental health reasoning** | 14-day longitudinal analysis; 4 clinical rule evaluators; 9 intent types |
| **Explainability** | Signal-level provenance; confidence scores; triggered data; full API explainability block |
| **Product thinking** | Multi-signal aggregation; API-first design; adaptive questioning; local-first architecture |
| **Code quality** | Modular architecture; dataclass-based data structures; graceful fallback; clean separation of layers |
| **UI quality** | Dark professional Streamlit dashboard; 4 pages; interactive charts; real-time LLM status |

---

## 9. Conclusion

VitaTwin demonstrates a complete, working implementation of all eight parts of the evaluation brief. The system uses a genuine local LLM (Ollama/llama3) with FAISS RAG for the conversational assistant, a multi-signal clinical rule engine with full signal-level explainability, a REST API exposing all required endpoints, and a polished Streamlit dashboard showing the complete pipeline in action.

The architecture is designed to be production-extensible: replacing TF-IDF with sentence transformers, SQLite with PostgreSQL, and Flask with FastAPI are all module-level changes. The clinical logic, LLM integration, explainability layer, and API contract remain unchanged.
