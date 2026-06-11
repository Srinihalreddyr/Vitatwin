"""
VitaTwin Clinical Intelligence API
Simulates a clinical REST API — no external services, 100% local.

Endpoints:
  POST /mental/screen       — screen a user for mental health risks
  POST /mental/ask          — ask a clinical question
  POST /mental/risk-score   — compute a user's risk score with explanation
  GET  /mental/summary/<id> — get emotional summary for a user
  GET  /mental/users        — list all users
  GET  /mental/health       — health check
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, request, jsonify
from models.clinical_engine import screen_user, ScreeningResult, Signal, Finding
from models.assistant import MentalHealthAssistant
from rag.rag_pipeline import retrieve_user_context, _load_index, retrieve_by_query

app = Flask(__name__)
assistant = MentalHealthAssistant()


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _serialize_signal(s: Signal) -> dict:
    return {
        "name":        s.name,
        "value":       s.value,
        "threshold":   s.threshold,
        "direction":   s.direction,
        "severity":    s.severity,
        "description": s.description,
    }

def _serialize_finding(f: Finding) -> dict:
    return {
        "category":          f.category,
        "label":             f.label,
        "risk_score":        f.risk_score,
        "confidence":        f.confidence,
        "signals":           [_serialize_signal(s) for s in f.signals],
        "explanation":       f.explanation,
        "suggested_actions": f.suggested_actions,
        "triggered_data":    f.triggered_data,
    }

def _serialize_result(r: ScreeningResult) -> dict:
    return {
        "user_id":           r.user_id,
        "overall_risk_score": r.overall_risk_score,
        "risk_level":        r.risk_level,
        "summary":           r.summary,
        "suggested_actions": r.suggested_actions,
        "confidence":        r.confidence,
        "raw_signals":       r.raw_signals,
        "findings":          [_serialize_finding(f) for f in r.findings],
    }

def _get_user_or_400(user_id: str):
    ctx = retrieve_user_context(user_id)
    if not ctx:
        return None, jsonify({"error": f"User {user_id} not found"}), 404
    return ctx["user"], None, None


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────

@app.route("/mental/health", methods=["GET"])
def health():
    _, meta = _load_index()
    return jsonify({
        "status":     "healthy",
        "service":    "VitaTwin Mental Health Intelligence API",
        "version":    "1.0.0",
        "users_indexed": len(meta["users"]),
        "rag_engine": "FAISS + TF-IDF (local)",
        "llm_engine": "VitaTwin Clinical Rule Engine (local)",
    })


@app.route("/mental/users", methods=["GET"])
def list_users():
    _, meta = _load_index()
    users = [
        {
            "user_id":    u["user_id"],
            "name":       u["name"],
            "age":        u["age"],
            "occupation": u["occupation"],
            "condition_label": u["condition_label"],
        }
        for u in meta["users"]
    ]
    return jsonify({"users": users, "total": len(users)})


@app.route("/mental/screen", methods=["POST"])
def screen():
    """
    POST /mental/screen
    Body: {"user_id": "VT001"}  OR  {"user_profile": {...}}
    Returns: full screening result with findings, signals, explanations.
    """
    body = request.get_json(force=True) or {}
    user_id = body.get("user_id")
    user_profile = body.get("user_profile")

    if user_id:
        user, err, code = _get_user_or_400(user_id)
        if err:
            return err, code
    elif user_profile:
        user = user_profile
    else:
        return jsonify({"error": "Provide user_id or user_profile"}), 400

    result = screen_user(user)
    return jsonify(_serialize_result(result))


@app.route("/mental/ask", methods=["POST"])
def ask():
    """
    POST /mental/ask
    Body: {"question": "...", "user_id": "VT001" (optional)}
    Returns: answer, risk context, RAG sources.
    """
    body = request.get_json(force=True) or {}
    question = body.get("question", "").strip()
    user_id  = body.get("user_id")

    if not question:
        return jsonify({"error": "question is required"}), 400

    resp = assistant.ask(question, user_id=user_id)
    return jsonify(resp)


@app.route("/mental/risk-score", methods=["POST"])
def risk_score():
    """
    POST /mental/risk-score
    Body: {"user_id": "VT001"}
    Returns: risk_score, risk_level, explanation, confidence, suggested_actions.
    """
    body = request.get_json(force=True) or {}
    user_id = body.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    user, err, code = _get_user_or_400(user_id)
    if err:
        return err, code

    result = screen_user(user)

    # Build per-finding explanations
    signal_details = []
    for f in result.findings:
        for s in f.signals:
            signal_details.append({
                "signal":      s.name,
                "description": s.description,
                "severity":    s.severity,
                "category":    f.category,
            })

    return jsonify({
        "user_id":      user_id,
        "name":         user["name"],
        "risk_score":   result.overall_risk_score,
        "risk_level":   result.risk_level,
        "confidence":   result.confidence,
        "explanation":  result.summary,
        "signal_details": signal_details,
        "suggested_actions": result.suggested_actions,
        "explainability": {
            "signals_used":     [s["signal"] for s in signal_details],
            "data_triggered":   result.raw_signals,
            "confidence_level": f"{result.confidence*100:.0f}%",
            "methodology":      "VitaTwin Clinical Rule Engine v1.0 — pattern-based multi-signal analysis",
        }
    })


@app.route("/mental/summary/<user_id>", methods=["GET"])
def summary(user_id):
    """
    GET /mental/summary/<user_id>
    Returns: emotional summary narrative + aggregated metrics.
    """
    user, err, code = _get_user_or_400(user_id)
    if err:
        return err, code

    result  = screen_user(user)
    resp    = assistant.ask("Summarize this user's emotional state", user_id=user_id)
    qs      = assistant.adaptive_questions(user_id)

    return jsonify({
        "user_id":   user_id,
        "name":      user["name"],
        "summary":   resp["answer"],
        "risk_level": result.risk_level,
        "risk_score": result.overall_risk_score,
        "metrics":   result.raw_signals,
        "adaptive_questions": qs,
        "suggested_actions":  result.suggested_actions[:3],
    })


@app.route("/mental/search", methods=["GET"])
def search():
    """GET /mental/search?q=stress+increasing — semantic user search"""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "q is required"}), 400
    results = retrieve_by_query(q, top_k=5)
    return jsonify({
        "query": q,
        "results": [
            {
                "user_id":   r["user"]["user_id"],
                "name":      r["user"]["name"],
                "condition": r["user"]["condition_label"],
                "similarity": r["similarity"],
            }
            for r in results
        ]
    })


if __name__ == "__main__":
    print("[API] Starting VitaTwin Mental Health API on http://localhost:5000")
    app.run(debug=False, port=5000)
