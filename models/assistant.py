"""
VitaTwin LLM Mental Health Assistant
Uses Ollama (local LLM) + FAISS RAG for genuine LLM-powered clinical responses.
Falls back to template engine if Ollama is not running.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag.rag_pipeline import retrieve_for_question, retrieve_user_context, _load_index
from models.clinical_engine import screen_user, ScreeningResult, _trend, _pct_change
from models.vector_memory import VectorMemory


# ──────────────────────────────────────────────────────────────
#  Ollama availability check
# ──────────────────────────────────────────────────────────────

def _ollama_available() -> bool:
    """Return True if Ollama is installed, running, and has at least one model."""
    try:
        import ollama  # noqa — optional dependency
        models = ollama.list()
        return len(models.get("models", [])) > 0
    except ImportError:
        return False   # ollama not installed (e.g. Streamlit Cloud)
    except Exception:
        return False   # ollama installed but not running


def _get_ollama_model() -> str:
    """Return the best available Ollama model name."""
    try:
        import ollama  # noqa
        models = ollama.list().get("models", [])
        names = [m["name"] for m in models]
        for preferred in ["llama3", "llama3:latest", "llama3.2", "llama3.1",
                          "mistral", "mistral:latest", "gemma2", "phi3",
                          "llama2", "llama2:latest"]:
            for n in names:
                if n.startswith(preferred):
                    return n
        return names[0] if names else "llama3"
    except Exception:
        return "llama3"


# ──────────────────────────────────────────────────────────────
#  Context builder  — converts raw user data into a rich text
#  block that the LLM can reason over
# ──────────────────────────────────────────────────────────────

def _build_user_context(user: dict, result: ScreeningResult) -> str:
    """
    Build a structured plain-text context block from the user's
    14-day longitudinal data + clinical engine findings.
    This is the RAG context injected into the LLM prompt.
    """
    agg = user["aggregates"]
    si  = user["social_indicators"]

    mood_vals   = [d["score"] for d in user["mood_history"]]
    stress_vals = [d["score"] for d in user["stress_scores"]]
    sleep_vals  = [d["hours"] for d in user["sleep_hours"]]
    energy_vals = [d["level"] for d in user["energy_levels"]]

    mood_slope   = _trend(mood_vals)
    stress_slope = _trend(stress_vals)
    sleep_slope  = _trend(sleep_vals)
    energy_slope = _trend(energy_vals)

    def trend_word(slope, invert=False):
        if invert:
            return "declining" if slope > 0.15 else "improving" if slope < -0.15 else "stable"
        return "rising" if slope > 0.15 else "falling" if slope < -0.15 else "stable"

    # Daily data as compact CSV-style string
    mood_series   = ", ".join(f"{d['score']}" for d in user["mood_history"])
    stress_series = ", ".join(f"{d['score']}" for d in user["stress_scores"])
    sleep_series  = ", ".join(f"{d['hours']}" for d in user["sleep_hours"])
    energy_series = ", ".join(f"{d['level']}" for d in user["energy_levels"])

    # Last 3 journal entries
    journals = "\n".join(
        f"  [{e['date']}]: {e['entry']}"
        for e in user["journal_entries"][-3:]
    )

    # Clinical findings from rule engine
    if result.findings:
        findings_text = "\n".join(
            f"  - {f.label} (score={f.risk_score:.0f}/100, confidence={f.confidence*100:.0f}%): "
            f"{'; '.join(s.description for s in f.signals[:4])}"
            for f in result.findings
        )
    else:
        findings_text = "  - No significant risk findings detected."

    suggested = "\n".join(f"  - {a}" for a in (result.suggested_actions or []))

    context = f"""
=== PATIENT PROFILE ===
Name: {user['name']} | ID: {user['user_id']} | Age: {user['age']}
Occupation: {user['occupation']}
Clinical condition label: {user['condition_label'].replace('_', ' ')}

=== 14-DAY AGGREGATE METRICS ===
Average Mood:   {agg['avg_mood_14d']:.1f}/10  (trend: {trend_word(mood_slope)})
Average Stress: {agg['avg_stress_14d']:.1f}/10  (trend: {trend_word(stress_slope)})
Average Sleep:  {agg['avg_sleep_14d']:.1f}h/night  (trend: {trend_word(sleep_slope, invert=True)})
Average Energy: {agg['avg_energy_14d']:.1f}/10  (trend: {trend_word(energy_slope)})

=== DAILY TIME-SERIES (14 days, day 1 → day 14) ===
Mood scores:   {mood_series}
Stress scores: {stress_series}
Sleep hours:   {sleep_series}
Energy levels: {energy_series}

=== SOCIAL & LIFESTYLE ===
Social interactions/week: {si['social_interactions_per_week']}
Exercise sessions/week:   {si['exercise_sessions_per_week']}
Work hours/day:           {si['work_hours_daily']:.1f}
Screen time/day:          {si['screen_time_hours_daily']:.1f}h
Meditation/day:           {si['meditation_minutes_daily']} min

=== RECENT JOURNAL ENTRIES ===
{journals}

=== CLINICAL ENGINE FINDINGS (rule-based signals) ===
Overall risk level: {result.risk_level.upper()} ({result.overall_risk_score:.0f}/100)
{findings_text}

=== SUGGESTED CLINICAL ACTIONS ===
{suggested if suggested else "  - Maintain current wellness habits"}
""".strip()

    return context


# ──────────────────────────────────────────────────────────────
#  LLM call via Ollama
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are VitaTwin, an expert clinical mental health analyst assistant.
You are given structured longitudinal data about a patient collected over 14 days, including mood, stress, sleep, energy scores, journal entries, and rule-based clinical findings.

Your role is to:
- Analyse the patient's mental health patterns from their data
- Provide clear, empathetic, evidence-based clinical insights
- Reference specific data points (scores, trends, journal entries) in your answers
- Be concise but thorough — 3 to 6 sentences is ideal unless a longer answer is needed
- Always speak in third person about the patient (e.g. "Alex shows..." not "you show...")
- Do not make diagnoses — frame everything as clinical observations and patterns
- Use the clinical engine findings as supporting evidence but reason beyond them

Respond in plain text with no JSON, no bullet lists unless asked, no markdown headers."""


def _ask_ollama(question: str, user_context: str, model: str) -> str:
    """Send a question + user context to Ollama and return the response text."""
    import ollama  # noqa — only called when _ollama_available() is True

    user_message = f"""Here is the patient data:

{user_context}

Clinical question: {question}"""

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        options={"temperature": 0.3, "num_predict": 512},
    )
    return response["message"]["content"].strip()


def _ask_ollama_adaptive_questions(user_context: str, model: str) -> list[str]:
    """Ask Ollama to generate adaptive check-in questions for this patient."""
    import ollama  # noqa

    prompt = f"""Here is the patient data:

{user_context}

Based on this patient's current mental health status and risk factors, generate exactly 3 short, empathetic, adaptive check-in questions a clinician would ask this specific patient today.
Return only the 3 questions, one per line, no numbering, no extra text."""

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "You are a clinical mental health assistant generating personalised check-in questions."},
            {"role": "user",   "content": prompt},
        ],
        options={"temperature": 0.4, "num_predict": 200},
    )
    text = response["message"]["content"].strip()
    questions = [q.strip() for q in text.split("\n") if q.strip()]
    return questions[:4]


# ──────────────────────────────────────────────────────────────
#  Template fallback (used when Ollama is not running)
# ──────────────────────────────────────────────────────────────

def _fallback_answer(question: str, user: dict, result: ScreeningResult) -> str:
    """Rule-based template response — used only when Ollama is unavailable."""
    agg = user["aggregates"]
    si  = user["social_indicators"]
    name = user["name"]

    mood_vals   = [d["score"] for d in user["mood_history"]]
    stress_vals = [d["score"] for d in user["stress_scores"]]
    sleep_vals  = [d["hours"] for d in user["sleep_hours"]]

    mood_slope   = _trend(mood_vals)
    stress_slope = _trend(stress_vals)
    sleep_slope  = _trend(sleep_vals)

    def tw(s, inv=False):
        if inv: return "declining" if s > 0.15 else "improving" if s < -0.15 else "stable"
        return "rising" if s > 0.15 else "falling" if s < -0.15 else "stable"

    q = question.lower()

    if any(w in q for w in ["risk", "concern", "flag", "danger", "warning"]):
        if not result.findings:
            return f"No significant mental health risks detected for {name}. All biomarkers fall within acceptable ranges."
        lines = [f"The following risks were identified for {name}:"]
        for f in result.findings:
            lines.append(f"• {f.label} (score: {f.risk_score:.0f}/100) — {f.explanation[:150]}")
        lines.append(f"\nOverall: {result.risk_level.upper()} RISK ({result.overall_risk_score:.0f}/100)")
        return "\n".join(lines)

    if any(w in q for w in ["summary", "summarize", "overall", "emotional state", "wellbeing"]):
        return (
            f"**{name} — 14-Day Emotional Summary**\n\n"
            f"Mood: {agg['avg_mood_14d']:.1f}/10 ({tw(mood_slope)}). "
            f"Stress: {agg['avg_stress_14d']:.1f}/10 ({tw(stress_slope)}). "
            f"Sleep: {agg['avg_sleep_14d']:.1f}h/night ({tw(sleep_slope, True)}). "
            f"Energy: {agg['avg_energy_14d']:.1f}/10.\n\n"
            f"Social: {si['social_interactions_per_week']}x/week. "
            f"Exercise: {si['exercise_sessions_per_week']}x/week. "
            f"Work: {si['work_hours_daily']:.1f}h/day.\n\n"
            f"Clinical assessment: **{result.risk_level.upper()} RISK** ({result.overall_risk_score:.0f}/100)."
        )

    if any(w in q for w in ["sleep"]):
        pct = _pct_change(sleep_vals, window=7)
        return (
            f"{name} averages {agg['avg_sleep_14d']:.1f}h/night over 14 days. "
            f"Sleep trend is {tw(sleep_slope, True)} "
            f"({'improving' if pct > 5 else 'worsening' if pct < -5 else 'stable'} by {abs(pct):.0f}% week-over-week)."
        )

    if any(w in q for w in ["stress"]):
        pct = _pct_change(stress_vals, window=7)
        return (
            f"{name}'s stress averages {agg['avg_stress_14d']:.1f}/10 over 14 days and is {tw(stress_slope)}. "
            f"Week-over-week change: {pct:+.0f}%. Latest reading: {stress_vals[-1]:.1f}/10."
        )

    if any(w in q for w in ["journal", "wrote", "diary"]):
        entries = user["journal_entries"][-3:]
        return "Recent journal entries:\n" + "\n".join(f"[{e['date']}]: {e['entry']}" for e in entries)

    if any(w in q for w in ["action", "recommend", "suggest", "help"]):
        actions = result.suggested_actions or ["Maintain current wellness habits"]
        return f"Recommended actions for {name}:\n" + "\n".join(f"• {a}" for a in actions)

    # General fallback
    return (
        f"{name} — {result.risk_level.upper()} RISK ({result.overall_risk_score:.0f}/100). "
        f"Mood: {agg['avg_mood_14d']:.1f}/10, Stress: {agg['avg_stress_14d']:.1f}/10, "
        f"Sleep: {agg['avg_sleep_14d']:.1f}h. "
        f"Primary concern: {result.findings[0].label if result.findings else 'none detected'}."
    )


def _fallback_adaptive_questions(user: dict, result: ScreeningResult) -> list[str]:
    """Template-based adaptive questions — fallback when Ollama unavailable."""
    questions = []
    agg = user["aggregates"]
    si  = user["social_indicators"]

    if result.risk_level in ("high", "critical"):
        questions.append("On a scale of 1–10, how would you rate your stress level today?")
        questions.append("Have you been able to get enough sleep this week?")
        questions.append("Is there anything specific that has been weighing on you lately?")
    elif result.risk_level == "moderate":
        questions.append("How has your energy been over the past few days?")
        questions.append("Have you had time for activities you enjoy recently?")
    else:
        questions.append("What has been going well for you this week?")
        questions.append("Is there anything you would like to focus on for your wellness?")

    if agg["avg_sleep_14d"] < 6:
        questions.append("Your sleep has been below ideal — what do you think is affecting it?")
    if si["social_interactions_per_week"] <= 2:
        questions.append("You've had fewer social interactions lately — how has that felt?")
    if agg["avg_stress_14d"] >= 7:
        questions.append("Your stress has been elevated — what strategies have been helping you cope?")

    return questions[:4]


# ──────────────────────────────────────────────────────────────
#  Main Assistant class
# ──────────────────────────────────────────────────────────────

class MentalHealthAssistant:

    def __init__(self):
        self._user_cache: dict = {}
        self._ollama_ok: bool = _ollama_available()
        self._model: str = _get_ollama_model() if self._ollama_ok else ""
        self.memory = VectorMemory()   # BONUS: vector memory for session recall
        if self._ollama_ok:
            print(f"[VitaTwin] Ollama available — using model: {self._model}")
        else:
            print("[VitaTwin] Ollama not running — using template fallback")

    def _get_user(self, user_id: str) -> dict | None:
        if user_id not in self._user_cache:
            ctx = retrieve_user_context(user_id)
            if ctx:
                self._user_cache[user_id] = ctx["user"]
        return self._user_cache.get(user_id)

    def _screen(self, user: dict) -> ScreeningResult:
        return screen_user(user)

    # ── public API ──

    def ask(self, question: str, user_id: str = None) -> dict:
        """
        Answer a clinical question. If user_id is given, scopes to that user.
        RAG retrieves context; Ollama (or fallback) generates the answer.
        """
        # ── Case 1: specific user ──
        if user_id:
            user = self._get_user(user_id)
            if not user:
                return {
                    "answer": f"User {user_id} not found in the system.",
                    "sources": [], "intent": "error", "user_id": user_id,
                }
            result = self._screen(user)

            if self._ollama_ok:
                try:
                    context = _build_user_context(user, result)
                    # BONUS: inject relevant memory turns into context
                    memory_ctx = self.memory.format_for_llm(question, user_id=user_id)
                    if memory_ctx:
                        context = memory_ctx + "\n\n" + context
                    answer  = _ask_ollama(question, context, self._model)
                except Exception as e:
                    answer = _fallback_answer(question, user, result)
                    answer += f"\n\n*(Ollama error: {e} — using fallback)*"
            else:
                answer = _fallback_answer(question, user, result)

            # BONUS: store this turn in vector memory
            self.memory.store(question, answer, user_id=user_id,
                              intent="llm" if self._ollama_ok else "template")

            return {
                "answer":        answer,
                "user_id":       user_id,
                "user_name":     user["name"],
                "intent":        "llm" if self._ollama_ok else "template",
                "risk_level":    result.risk_level,
                "risk_score":    result.overall_risk_score,
                "sources":       [f"VitaTwin RAG profile {user_id}"],
                "llm_used":      self._ollama_ok,
                "model":         self._model if self._ollama_ok else "template-engine",
                "memory_turns":  self.memory.size,
            }

        # ── Case 2: semantic search across all users ──
        rag_results = retrieve_for_question(question, top_k=3)
        hits = rag_results["similar_users"]

        if not hits:
            return {"answer": "No relevant user data found for that query.",
                    "sources": [], "intent": "no_results"}

        top_user = hits[0]["user"]
        result   = self._screen(top_user)

        if self._ollama_ok:
            try:
                context = _build_user_context(top_user, result)
                answer  = _ask_ollama(question, context, self._model)
            except Exception as e:
                answer = _fallback_answer(question, top_user, result)
                answer += f"\n\n*(Ollama error: {e} — using fallback)*"
        else:
            answer = _fallback_answer(question, top_user, result)

        answer += f"\n\n*Most relevant profile: {top_user['name']} ({top_user['user_id']}) — similarity {hits[0]['similarity']:.3f}*"

        return {
            "answer":     answer,
            "intent":     "llm" if self._ollama_ok else "template",
            "top_match":  top_user["user_id"],
            "risk_level": result.risk_level,
            "risk_score": result.overall_risk_score,
            "sources":    [f"{r['user']['user_id']}: {r['user']['name']} (sim={r['similarity']:.3f})" for r in hits],
            "llm_used":   self._ollama_ok,
            "model":      self._model if self._ollama_ok else "template-engine",
        }

    def adaptive_questions(self, user_id: str) -> list[str]:
        """Generate LLM-powered adaptive check-in questions for a specific user."""
        user = self._get_user(user_id)
        if not user:
            return ["How have you been feeling lately?"]

        result = self._screen(user)

        if self._ollama_ok:
            try:
                context   = _build_user_context(user, result)
                questions = _ask_ollama_adaptive_questions(context, self._model)
                if questions:
                    return questions
            except Exception:
                pass  # fall through to template

        return _fallback_adaptive_questions(user, result)

    def converse(self, messages: list[dict], user_id: str = None) -> dict:
        """Multi-turn conversation support."""
        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        return self.ask(last_user_msg, user_id=user_id)

    @property
    def llm_status(self) -> str:
        """Human-readable status for display in the UI."""
        if self._ollama_ok:
            return f"🟢 Ollama · {self._model}"
        return "🟡 Template fallback (Ollama not running)"

    @property
    def memory_status(self) -> str:
        """Human-readable vector memory status for display in the UI."""
        s = self.memory.summary()
        return f"🧠 Vector Memory · {s['turns_stored']} turns · vocab {s['vocab_size']}"


    def start_checkin(self, user_id: str) -> dict:
        """
        Begin a structured conversational check-in for a user.
        The assistant greets the user, summarises their recent emotional
        pattern, and asks the first adaptive question — opening the conversation.
        """
        user = self._get_user(user_id)
        if not user:
            return {"message": "User not found.", "next_question": None, "stage": "error"}

        result  = self._screen(user)
        agg     = user["aggregates"]
        name    = user["name"].split()[0]   # first name only

        if self._ollama_ok:
            try:
                context  = _build_user_context(user, result)
                prompt   = (
                    f"You are starting a supportive mental health check-in conversation with {name}. "
                    f"In 2-3 warm, empathetic sentences: greet them by first name, briefly mention one "
                    f"key emotional pattern you notice in their recent data (e.g. stress trend, sleep quality), "
                    f"then ask ONE gentle open-ended check-in question to start the conversation. "
                    f"Do NOT list multiple questions. Be warm, not clinical."
                )
                greeting = _ask_ollama(prompt, context, self._model)
            except Exception:
                greeting = self._template_greeting(name, agg, result)
        else:
            greeting = self._template_greeting(name, agg, result)

        self.memory.store(
            question = f"[CHECK-IN START for {user_id}]",
            answer   = greeting,
            user_id  = user_id,
            intent   = "checkin_start",
        )

        return {
            "message":       greeting,
            "stage":         "checkin",
            "user_name":     name,
            "risk_level":    result.risk_level,
            "risk_score":    result.overall_risk_score,
        }

    def reply_to_checkin(self, user_response: str, user_id: str,
                         conversation: list[dict]) -> dict:
        """
        Continue a conversational check-in.
        Receives what the user said, responds supportively, then either
        asks a follow-up adaptive question or wraps up with a summary.
        conversation = list of {"role": "user"|"assistant", "content": str}
        """
        user   = self._get_user(user_id)
        result = self._screen(user)
        name   = user["name"].split()[0]
        turn   = sum(1 for m in conversation if m["role"] == "user")

        if self._ollama_ok:
            try:
                context = _build_user_context(user, result)
                sep = "\n"
                history = sep.join(
                    f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                    for m in conversation[-6:]
                )
                memory_ctx = self.memory.format_for_llm(user_response, user_id=user_id)

                nl = "\n"
                if turn >= 3:
                    prompt = (
                        f"This is the end of a check-in with {name}. "
                        f"Based on the conversation and their clinical data, write a warm 3-4 sentence "
                        f"closing message that: (1) acknowledges what they shared, "
                        f"(2) summarises their emotional pattern over the past two weeks in plain language, "
                        f"(3) offers one practical supportive suggestion, "
                        f"(4) encourages them to reach out to a professional if needed. "
                        f"Be warm and human, not clinical.{nl}{nl}"
                        f"Conversation so far:{nl}{history}"
                    )
                    stage = "summary"
                else:
                    prompt = (
                        f"You are in a supportive mental health check-in with {name} (turn {turn}).{nl}"
                        f"Conversation so far:{nl}{history}{nl}{nl}"
                        f"Their latest message: '{user_response}'{nl}{nl}"
                        f"In 2-3 sentences: (1) respond with empathy and validation to what they just said, "
                        f"(2) ask ONE gentle follow-up question to go deeper. "
                        f"Do NOT give advice yet. Do NOT ask multiple questions. Be warm and human."
                    )
                    stage = "checkin"

                if memory_ctx:
                    context = memory_ctx + "\n\n" + context

                reply = _ask_ollama(prompt, context, self._model)

            except Exception as e:
                reply = self._template_reply(user_response, name, result, turn)
                stage = "summary" if turn >= 3 else "checkin"
        else:
            reply = self._template_reply(user_response, name, result, turn)
            stage = "summary" if turn >= 3 else "checkin"

        self.memory.store(
            question = user_response,
            answer   = reply,
            user_id  = user_id,
            intent   = "checkin_reply",
        )

        return {
            "message":    reply,
            "stage":      stage,
            "risk_level": result.risk_level,
            "risk_score": result.overall_risk_score,
        }

    # ── template helpers for when Ollama is offline ────────────

    def _template_greeting(self, name: str, agg: dict,
                            result: ScreeningResult) -> str:
        stress = agg["avg_stress_14d"]
        sleep  = agg["avg_sleep_14d"]
        mood   = agg["avg_mood_14d"]

        if stress >= 7:
            obs = f"I've noticed your stress levels have been quite elevated recently — averaging {stress:.1f}/10 over the past two weeks."
        elif sleep < 6:
            obs = f"I've noticed your sleep has been lower than ideal lately, averaging around {sleep:.1f} hours per night."
        elif mood <= 4:
            obs = f"I've noticed your mood scores have been on the lower side recently, averaging {mood:.1f}/10."
        else:
            obs = f"Your recent wellbeing data looks generally stable, which is good to see."

        return (
            f"Hi {name}, I'm VitaTwin — here to check in with you today. 😊 "
            f"{obs} "
            f"I'd love to hear how you're actually feeling — in your own words, how has this past week been for you?"
        )

    def _template_reply(self, user_response: str, name: str,
                        result: ScreeningResult, turn: int) -> str:
        if turn >= 3:
            findings = result.findings[0].label if result.findings else "some stress indicators"
            return (
                f"Thank you so much for sharing that with me, {name}. "
                f"Looking at your recent data alongside what you've told me, I can see that {findings.lower()} "
                f"has been a pattern over the past two weeks — and what you've described today reflects that. "
                f"It's really important that you're acknowledging how you feel. "
                f"One thing that might help is setting aside even 10 minutes a day for something restorative — "
                f"a walk, journaling, or simply stepping away from screens. "
                f"If things feel overwhelming, please do reach out to a mental health professional. You deserve support. 💙"
            )

        follow_ups = [
            f"That sounds really tough, {name}. What do you think has been contributing to that the most?",
            f"I hear you. How has that been affecting your sleep or energy levels day-to-day?",
            f"Thank you for sharing that. On your better days, what tends to help you feel more grounded?",
        ]
        return (
            f"Thank you for sharing that, {name} — it takes courage to reflect on how we're feeling. "
            + follow_ups[min(turn - 1, len(follow_ups) - 1)]
        )


# ──────────────────────────────────────────────────────────────
#  CLI test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    assistant = MentalHealthAssistant()
    print(f"\nLLM status: {assistant.llm_status}\n")

    print("=" * 60)
    print("TEST 1: Summarize emotional state [VT001]")
    resp = assistant.ask("Summarize this user's emotional state", user_id="VT001")
    print(resp["answer"])
    print(f"[model: {resp['model']}]")

    print("\n" + "=" * 60)
    print("TEST 2: What mental health risks are visible? [VT001]")
    resp = assistant.ask("What mental health risks are visible?", user_id="VT001")
    print(resp["answer"])

    print("\n" + "=" * 60)
    print("TEST 3: Adaptive questions [VT001]")
    for q in assistant.adaptive_questions("VT001"):
        print(f"  • {q}")
