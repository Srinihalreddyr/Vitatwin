"""
VitaTwin Clinical Intelligence Layer
Rule-based early detection engine modelled after real clinical decision support.
Every finding carries: signals used, data that triggered it, confidence level.
"""

from dataclasses import dataclass, field
from typing import Optional
import statistics


# ─────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────

@dataclass
class Signal:
    name: str
    value: float
    threshold: float
    direction: str          # "high" | "low" | "trend_up" | "trend_down"
    severity: str           # "mild" | "moderate" | "severe"
    description: str


@dataclass
class Finding:
    category: str           # e.g. "burnout_risk"
    label: str              # human-readable label
    risk_score: float       # 0–100
    confidence: float       # 0–1
    signals: list[Signal]
    explanation: str
    suggested_actions: list[str]
    triggered_data: dict    # raw values that fired the rules


@dataclass
class ScreeningResult:
    user_id: str
    overall_risk_score: float
    risk_level: str         # low / moderate / high / critical
    findings: list[Finding]
    summary: str
    suggested_actions: list[str]
    confidence: float
    raw_signals: dict


# ─────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────

def _trend(series: list[float], window: int = 7) -> float:
    """Simple linear trend slope over the last `window` points."""
    if len(series) < 2:
        return 0.0
    recent = series[-window:]
    n = len(recent)
    x_mean = (n - 1) / 2
    y_mean = statistics.mean(recent)
    num = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n)) or 1e-9
    return num / den


def _pct_change(series: list[float], window: int = 7) -> float:
    """% change from `window` days ago to now."""
    if len(series) < window:
        return 0.0
    old = series[-window]
    new = series[-1]
    if old == 0:
        return 0.0
    return (new - old) / old * 100


def _negative_journal_ratio(entries: list[dict]) -> float:
    """Rough sentiment: fraction of recent journals with negative keywords."""
    NEG = {"overwhelmed","drained","anxious","stressed","exhausted","depressed",
           "hopeless","tired","crying","isolated","negative","worst","terrible",
           "horrible","headache","tension","pointless","ruminating","withdrawn",
           "snapped","worry","worried","anxious","anxiety","panic","numb"}
    recent = entries[-7:]
    hits = 0
    for e in recent:
        tokens = set(e["entry"].lower().split())
        if tokens & NEG:
            hits += 1
    return hits / max(len(recent), 1)


# ─────────────────────────────────────────────
#  Rule evaluators
# ─────────────────────────────────────────────

def _eval_burnout(user: dict) -> Optional[Finding]:
    agg = user["aggregates"]
    si  = user["social_indicators"]
    stress_series = [d["score"] for d in user["stress_scores"]]
    sleep_series  = [d["hours"] for d in user["sleep_hours"]]
    energy_series = [d["level"] for d in user["energy_levels"]]
    mood_series   = [d["score"] for d in user["mood_history"]]

    stress_trend = _trend(stress_series)
    sleep_pct    = _pct_change(sleep_series)
    energy_avg   = agg["avg_energy_14d"]
    work_hrs     = si["work_hours_daily"]
    neg_ratio    = _negative_journal_ratio(user["journal_entries"])

    signals = []
    score = 0.0

    if agg["avg_stress_14d"] >= 7.0:
        sev = "severe" if agg["avg_stress_14d"] >= 8.5 else "moderate"
        signals.append(Signal("avg_stress", agg["avg_stress_14d"], 7.0, "high", sev,
                               f"14-day avg stress = {agg['avg_stress_14d']:.1f}/10"))
        score += 25 if sev == "severe" else 15

    if stress_trend > 0.15:
        signals.append(Signal("stress_trend", stress_trend, 0.15, "trend_up", "moderate",
                               f"Stress rising at +{stress_trend:.2f}/day over last 7 days"))
        score += 15

    if sleep_pct <= -20:
        signals.append(Signal("sleep_decline", abs(sleep_pct), 20, "trend_down", "moderate",
                               f"Sleep declined {abs(sleep_pct):.0f}% over last 7 days"))
        score += 20

    if energy_avg <= 4.0:
        sev = "severe" if energy_avg <= 2.5 else "moderate"
        signals.append(Signal("low_energy", energy_avg, 4.0, "low", sev,
                               f"14-day avg energy = {energy_avg:.1f}/10"))
        score += 20 if sev == "severe" else 10

    if work_hrs >= 10:
        signals.append(Signal("overwork", work_hrs, 10.0, "high", "mild",
                               f"Daily work hours = {work_hrs:.1f}h"))
        score += 10

    if neg_ratio >= 0.5:
        sev = "severe" if neg_ratio >= 0.75 else "moderate"
        signals.append(Signal("negative_journals", neg_ratio, 0.5, "high", sev,
                               f"{neg_ratio*100:.0f}% of recent journals contain negative sentiment"))
        score += 20 if sev == "severe" else 10

    if not signals:
        return None

    score = min(score, 100)
    confidence = min(len(signals) / 5.0, 1.0)

    actions = ["Recommend immediate workload reduction",
               "Suggest structured daily rest periods",
               "Introduce mindfulness or stress-reduction techniques",
               "Flag for proactive clinical check-in within 48 hours"]
    if score < 40:
        actions = ["Monitor stress levels daily",
                   "Encourage healthy work-life boundaries",
                   "Suggest weekly wellness check-in"]

    explanation = (
        f"Burnout risk detected based on {len(signals)} signal(s). "
        + " ".join(s.description + "." for s in signals)
    )

    return Finding(
        category="burnout_risk",
        label="Potential Burnout Risk",
        risk_score=round(score, 1),
        confidence=round(confidence, 2),
        signals=signals,
        explanation=explanation,
        suggested_actions=actions,
        triggered_data={
            "avg_stress_14d": agg["avg_stress_14d"],
            "stress_trend_slope": round(stress_trend, 3),
            "sleep_pct_change_7d": round(sleep_pct, 1),
            "avg_energy_14d": energy_avg,
            "work_hours_daily": work_hrs,
            "negative_journal_ratio": round(neg_ratio, 2),
        }
    )


def _eval_anxiety(user: dict) -> Optional[Finding]:
    agg = user["aggregates"]
    stress_series = [d["score"] for d in user["stress_scores"]]
    sleep_series  = [d["hours"] for d in user["sleep_hours"]]
    neg_ratio     = _negative_journal_ratio(user["journal_entries"])
    stress_variability = statistics.stdev(stress_series) if len(stress_series) > 1 else 0

    signals = []
    score = 0.0

    if agg["avg_stress_14d"] >= 6.5:
        signals.append(Signal("elevated_stress", agg["avg_stress_14d"], 6.5, "high", "moderate",
                               f"Persistent elevated stress = {agg['avg_stress_14d']:.1f}/10"))
        score += 20

    if stress_variability >= 1.8:
        signals.append(Signal("stress_volatility", stress_variability, 1.8, "high", "moderate",
                               f"High stress volatility (σ={stress_variability:.2f}) suggests anxiety spikes"))
        score += 20

    if agg["avg_sleep_14d"] < 6.0:
        signals.append(Signal("poor_sleep", agg["avg_sleep_14d"], 6.0, "low", "moderate",
                               f"Insufficient sleep avg = {agg['avg_sleep_14d']:.1f}h"))
        score += 20

    if neg_ratio >= 0.4:
        signals.append(Signal("negative_journals", neg_ratio, 0.4, "high", "mild",
                               f"{neg_ratio*100:.0f}% negative journal entries"))
        score += 15

    si = user["social_indicators"]
    if si["social_interactions_per_week"] <= 2:
        signals.append(Signal("social_withdrawal", si["social_interactions_per_week"], 2, "low", "mild",
                               f"Social interactions = {si['social_interactions_per_week']}/week (low)"))
        score += 10

    if not signals:
        return None

    score = min(score, 100)
    confidence = min(len(signals) / 4.0, 1.0)

    explanation = (
        f"Anxiety indicators present across {len(signals)} domain(s). "
        + " ".join(s.description + "." for s in signals)
    )
    actions = [
        "Recommend cognitive behavioural techniques for anxiety",
        "Suggest sleep hygiene improvements",
        "Encourage social re-engagement activities",
        "Consider referral to mental health professional if score > 60",
    ]

    return Finding(
        category="anxiety_trend",
        label="Anxiety Pattern Detected",
        risk_score=round(score, 1),
        confidence=round(confidence, 2),
        signals=signals,
        explanation=explanation,
        suggested_actions=actions,
        triggered_data={
            "avg_stress_14d": agg["avg_stress_14d"],
            "stress_volatility": round(stress_variability, 2),
            "avg_sleep_14d": agg["avg_sleep_14d"],
            "negative_journal_ratio": round(neg_ratio, 2),
            "social_interactions_week": si["social_interactions_per_week"],
        }
    )


def _eval_depression(user: dict) -> Optional[Finding]:
    agg = user["aggregates"]
    si  = user["social_indicators"]
    mood_series = [d["score"] for d in user["mood_history"]]
    mood_trend  = _trend(mood_series)
    neg_ratio   = _negative_journal_ratio(user["journal_entries"])

    signals = []
    score = 0.0

    if agg["avg_mood_14d"] <= 4.0:
        sev = "severe" if agg["avg_mood_14d"] <= 2.5 else "moderate"
        signals.append(Signal("low_mood", agg["avg_mood_14d"], 4.0, "low", sev,
                               f"14-day avg mood = {agg['avg_mood_14d']:.1f}/10"))
        score += 25 if sev == "severe" else 15

    if mood_trend < -0.15:
        signals.append(Signal("declining_mood", abs(mood_trend), 0.15, "trend_down", "moderate",
                               f"Mood declining at {mood_trend:.2f}/day over last 7 days"))
        score += 20

    if si["exercise_sessions_per_week"] == 0:
        signals.append(Signal("no_exercise", 0, 1, "low", "mild",
                               "Zero exercise sessions per week reported"))
        score += 10

    if si["social_interactions_per_week"] <= 1:
        signals.append(Signal("isolation", si["social_interactions_per_week"], 1, "low", "moderate",
                               f"Near-zero social contact ({si['social_interactions_per_week']}/week)"))
        score += 20

    if agg["avg_energy_14d"] <= 3.0:
        signals.append(Signal("very_low_energy", agg["avg_energy_14d"], 3.0, "low", "severe",
                               f"Very low energy = {agg['avg_energy_14d']:.1f}/10"))
        score += 20

    if neg_ratio >= 0.6:
        signals.append(Signal("persistent_negative_cognition", neg_ratio, 0.6, "high", "severe",
                               f"{neg_ratio*100:.0f}% of journals show persistent negative thought patterns"))
        score += 25

    if not signals:
        return None

    score = min(score, 100)
    confidence = min(len(signals) / 5.0, 1.0)

    explanation = (
        f"Depression indicators detected across {len(signals)} signal(s). "
        + " ".join(s.description + "." for s in signals)
    )
    actions = [
        "PRIORITY: Immediate referral to mental health clinician recommended",
        "Conduct PHQ-9 assessment within 24 hours",
        "Enable safety monitoring protocols",
        "Encourage behavioural activation starting with small achievable tasks",
        "Daily check-in calls recommended",
    ]

    return Finding(
        category="depression_indicators",
        label="Depression Indicators Detected",
        risk_score=round(score, 1),
        confidence=round(confidence, 2),
        signals=signals,
        explanation=explanation,
        suggested_actions=actions,
        triggered_data={
            "avg_mood_14d": agg["avg_mood_14d"],
            "mood_trend_slope": round(mood_trend, 3),
            "exercise_per_week": si["exercise_sessions_per_week"],
            "social_interactions_week": si["social_interactions_per_week"],
            "avg_energy_14d": agg["avg_energy_14d"],
            "negative_journal_ratio": round(neg_ratio, 2),
        }
    )


def _eval_resilience(user: dict) -> Optional[Finding]:
    agg = user["aggregates"]
    si  = user["social_indicators"]
    if (agg["avg_mood_14d"] >= 7 and agg["avg_stress_14d"] <= 4 and
            agg["avg_sleep_14d"] >= 7 and si["exercise_sessions_per_week"] >= 3):
        signals = [
            Signal("high_mood", agg["avg_mood_14d"], 7.0, "high", "mild",
                   f"Consistent high mood avg = {agg['avg_mood_14d']:.1f}/10"),
            Signal("low_stress", agg["avg_stress_14d"], 4.0, "low", "mild",
                   f"Well-managed stress avg = {agg['avg_stress_14d']:.1f}/10"),
        ]
        return Finding(
            category="resilient",
            label="Strong Resilience Indicators",
            risk_score=0.0,
            confidence=0.85,
            signals=signals,
            explanation="User demonstrates strong mental wellness markers across mood, stress, sleep, and activity.",
            suggested_actions=["Maintain current wellness practices",
                                "Share resilience strategies in group sessions"],
            triggered_data={
                "avg_mood_14d": agg["avg_mood_14d"],
                "avg_stress_14d": agg["avg_stress_14d"],
                "avg_sleep_14d": agg["avg_sleep_14d"],
                "exercise_per_week": si["exercise_sessions_per_week"],
            }
        )
    return None


# ─────────────────────────────────────────────
#  Main screener
# ─────────────────────────────────────────────

def screen_user(user: dict) -> ScreeningResult:
    findings = []
    for evaluator in [_eval_burnout, _eval_anxiety, _eval_depression, _eval_resilience]:
        f = evaluator(user)
        if f:
            findings.append(f)

    if findings:
        # Overall score = max of all finding scores (not avg — worst risk drives the headline)
        risk_findings = [f for f in findings if f.category != "resilient"]
        overall = max((f.risk_score for f in risk_findings), default=0.0)
    else:
        overall = 0.0

    if overall == 0 and any(f.category == "resilient" for f in findings):
        risk_level = "low"
    elif overall <= 30:
        risk_level = "low"
    elif overall <= 55:
        risk_level = "moderate"
    elif overall <= 75:
        risk_level = "high"
    else:
        risk_level = "critical"

    all_actions = []
    seen = set()
    for f in findings:
        for a in f.suggested_actions:
            if a not in seen:
                all_actions.append(a)
                seen.add(a)

    # Build summary narrative
    if not findings:
        summary = (f"User {user['name']} shows no significant mental health risk indicators. "
                   f"All biomarkers are within normal ranges.")
    else:
        top = sorted(findings, key=lambda x: -x.risk_score)
        labels = ", ".join(f.label for f in top[:2])
        summary = (
            f"Over the past 14 days, {user['name']} shows {labels.lower()}. "
            f"Key signals include: "
            + (top[0].signals[0].description if top[0].signals else "multiple indicators")
            + ". "
            f"Overall risk level is {risk_level.upper()} (score {overall:.0f}/100)."
        )

    avg_conf = round(sum(f.confidence for f in findings) / max(len(findings), 1), 2)

    return ScreeningResult(
        user_id=user["user_id"],
        overall_risk_score=round(overall, 1),
        risk_level=risk_level,
        findings=findings,
        summary=summary,
        suggested_actions=all_actions[:5],
        confidence=avg_conf,
        raw_signals={
            "avg_mood_14d":   user["aggregates"]["avg_mood_14d"],
            "avg_stress_14d": user["aggregates"]["avg_stress_14d"],
            "avg_sleep_14d":  user["aggregates"]["avg_sleep_14d"],
            "avg_energy_14d": user["aggregates"]["avg_energy_14d"],
            "work_hours_daily": user["social_indicators"]["work_hours_daily"],
            "exercise_per_week": user["social_indicators"]["exercise_sessions_per_week"],
            "social_interactions_week": user["social_indicators"]["social_interactions_per_week"],
        }
    )


if __name__ == "__main__":
    import json, os
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "users.json")) as f:
        users = json.load(f)
    # Test on a few users
    for u in users[:5]:
        r = screen_user(u)
        print(f"\n{r.user_id} | {u['name']} | {r.risk_level.upper()} | score={r.overall_risk_score}")
        for fi in r.findings:
            print(f"  [{fi.category}] {fi.label} — conf={fi.confidence}")
            print(f"    {fi.explanation[:120]}…")
