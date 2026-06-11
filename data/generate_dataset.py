"""
VitaTwin Dataset Generator
Generates 50 synthetic mental health user profiles with realistic patterns.
"""

import json
import csv
import random
import math
import os
from datetime import datetime, timedelta

random.seed(42)

NAMES = [
    "Alex Chen","Jordan Rivera","Sam Patel","Morgan Lee","Casey Kim",
    "Riley Thompson","Taylor Nguyen","Drew Martinez","Cameron Singh","Avery Johnson",
    "Blake Williams","Emery Davis","Finley Brown","Harper Wilson","Indigo Garcia",
    "Jamie Anderson","Kennedy Thomas","Logan Jackson","Mason White","Nat Harris",
    "Olivia Martin","Parker Robinson","Quinn Clark","Rowan Lewis","Sage Walker",
    "Skylar Hall","Sterling Allen","Tatum Young","Umber Hernandez","Vale King",
    "Wren Wright","Xander Scott","Yael Green","Zara Baker","Adrian Adams",
    "Briar Nelson","Cedar Carter","Demi Mitchell","Evan Perez","Faye Roberts",
    "Gray Turner","Haven Phillips","Iris Campbell","Juno Parker","Kai Evans",
    "Lake Edwards","Miles Collins","Nova Stewart","Ocean Morris","Piper Rogers"
]

CONDITIONS = [
    "healthy", "mild_stress", "moderate_stress", "severe_stress",
    "burnout_risk", "anxiety_trend", "depression_indicators", "resilient"
]

OCCUPATIONS = [
    "Software Engineer","Medical Resident","Graduate Student","Teacher",
    "Marketing Manager","Nurse","Freelance Designer","Financial Analyst",
    "Social Worker","Data Scientist","Project Manager","Researcher",
    "Consultant","Artist","Entrepreneur","Journalist","Lawyer","Chef"
]

POSITIVE_JOURNAL = [
    "Had a productive day. Finished the project I was working on.",
    "Went for a long walk. Feeling refreshed and calm.",
    "Caught up with an old friend. It was really uplifting.",
    "Good sleep last night. Energy levels are great today.",
    "Meditation session helped. Mind feels clearer.",
    "Accomplished my goals for the week. Proud of myself.",
    "Family dinner was wonderful. Feeling connected.",
    "Exercised this morning. Body and mind feel aligned.",
    "Completed a creative project I enjoy. Very fulfilling.",
    "Took a mental health day. It was exactly what I needed.",
]

NEUTRAL_JOURNAL = [
    "Average day. Nothing remarkable happened.",
    "Busy with work but managing okay.",
    "Had some ups and downs but overall okay.",
    "Routine day. Met deadlines, nothing special.",
    "Feeling okay. Neither great nor bad.",
    "Work was demanding but I handled it.",
    "Tired after a long day but fine overall.",
    "Mixed feelings today. Hard to pinpoint why.",
    "The week has been steady, nothing extreme.",
    "Managing everything, just a bit worn.",
]

NEGATIVE_JOURNAL = [
    "Couldn't focus. Brain fog all day. Worried about deadlines.",
    "Slept badly again. Keep waking up at 3am with anxiety.",
    "Overwhelmed by everything. Don't know where to start.",
    "Feel completely drained. No motivation to do anything.",
    "Stress is through the roof. Snapped at a colleague today.",
    "Can't stop ruminating. Negative thoughts won't quiet down.",
    "Missed another workout. Just too exhausted to care.",
    "Feeling isolated. Haven't really talked to anyone meaningful.",
    "Another late night. Sacrificed sleep for work again.",
    "Headache and tension all day. Body is showing the stress.",
    "Everything feels pointless today. Very low energy.",
    "Anxiety spike this morning. Heart was racing for no reason.",
    "Crying without a clear reason. Emotional exhaustion kicking in.",
    "Withdrew from social plans again. Don't feel like seeing anyone.",
]

def generate_mood_history(condition, days=14):
    base = {"healthy": 7.5, "mild_stress": 6.0, "moderate_stress": 4.5,
            "severe_stress": 3.0, "burnout_risk": 3.5, "anxiety_trend": 4.0,
            "depression_indicators": 2.5, "resilient": 8.0}
    trend = {"healthy": 0.0, "mild_stress": -0.05, "moderate_stress": -0.1,
             "severe_stress": -0.15, "burnout_risk": -0.2, "anxiety_trend": -0.1,
             "depression_indicators": -0.25, "resilient": 0.05}
    base_val = base.get(condition, 5.0)
    t = trend.get(condition, 0)
    history = []
    for i in range(days):
        date = (datetime.now() - timedelta(days=days - i)).strftime("%Y-%m-%d")
        val = max(1, min(10, base_val + t * i + random.gauss(0, 0.5)))
        history.append({"date": date, "score": round(val, 1)})
    return history

def generate_stress_scores(condition, days=14):
    base = {"healthy": 3.0, "mild_stress": 5.0, "moderate_stress": 6.5,
            "severe_stress": 8.5, "burnout_risk": 8.0, "anxiety_trend": 7.5,
            "depression_indicators": 6.0, "resilient": 2.5}
    trend = {"healthy": 0.0, "mild_stress": 0.1, "moderate_stress": 0.15,
             "severe_stress": 0.2, "burnout_risk": 0.25, "anxiety_trend": 0.2,
             "depression_indicators": 0.05, "resilient": -0.05}
    base_val = base.get(condition, 5.0)
    t = trend.get(condition, 0)
    scores = []
    for i in range(days):
        date = (datetime.now() - timedelta(days=days - i)).strftime("%Y-%m-%d")
        val = max(1, min(10, base_val + t * i + random.gauss(0, 0.4)))
        scores.append({"date": date, "score": round(val, 1)})
    return scores

def generate_sleep_hours(condition, days=14):
    base = {"healthy": 7.5, "mild_stress": 7.0, "moderate_stress": 6.0,
            "severe_stress": 5.0, "burnout_risk": 5.5, "anxiety_trend": 5.5,
            "depression_indicators": 6.5, "resilient": 8.0}
    trend = {"healthy": 0.0, "mild_stress": -0.05, "moderate_stress": -0.08,
             "severe_stress": -0.12, "burnout_risk": -0.15, "anxiety_trend": -0.1,
             "depression_indicators": 0.05, "resilient": 0.02}
    base_val = base.get(condition, 7.0)
    t = trend.get(condition, 0)
    records = []
    for i in range(days):
        date = (datetime.now() - timedelta(days=days - i)).strftime("%Y-%m-%d")
        val = max(3, min(10, base_val + t * i + random.gauss(0, 0.4)))
        records.append({"date": date, "hours": round(val, 1)})
    return records

def generate_energy_levels(condition, days=14):
    base = {"healthy": 7.5, "mild_stress": 6.5, "moderate_stress": 5.0,
            "severe_stress": 3.5, "burnout_risk": 2.5, "anxiety_trend": 4.5,
            "depression_indicators": 2.0, "resilient": 8.5}
    base_val = base.get(condition, 5.0)
    records = []
    for i in range(days):
        date = (datetime.now() - timedelta(days=days - i)).strftime("%Y-%m-%d")
        val = max(1, min(10, base_val + random.gauss(0, 0.6)))
        records.append({"date": date, "level": round(val, 1)})
    return records

def generate_journal_entries(condition, days=14):
    if condition in ["healthy", "resilient"]:
        pool = POSITIVE_JOURNAL * 2 + NEUTRAL_JOURNAL
    elif condition == "mild_stress":
        pool = NEUTRAL_JOURNAL * 2 + POSITIVE_JOURNAL + NEGATIVE_JOURNAL
    elif condition in ["moderate_stress", "anxiety_trend"]:
        pool = NEUTRAL_JOURNAL + NEGATIVE_JOURNAL * 2
    else:
        pool = NEGATIVE_JOURNAL * 3 + NEUTRAL_JOURNAL
    entries = []
    for i in range(days):
        date = (datetime.now() - timedelta(days=days - i)).strftime("%Y-%m-%d")
        entries.append({"date": date, "entry": random.choice(pool)})
    return entries

def generate_social_indicators(condition):
    social_base = {"healthy": 7, "mild_stress": 6, "moderate_stress": 4,
                   "severe_stress": 2, "burnout_risk": 2, "anxiety_trend": 3,
                   "depression_indicators": 1, "resilient": 8}
    exercise_base = {"healthy": 4, "mild_stress": 3, "moderate_stress": 2,
                     "severe_stress": 1, "burnout_risk": 1, "anxiety_trend": 2,
                     "depression_indicators": 1, "resilient": 5}
    return {
        "social_interactions_per_week": max(0, social_base.get(condition, 4) + random.randint(-1, 1)),
        "exercise_sessions_per_week": max(0, exercise_base.get(condition, 2) + random.randint(-1, 1)),
        "screen_time_hours_daily": round(random.uniform(4, 10), 1),
        "work_hours_daily": round(random.uniform(7, 14), 1),
        "meditation_minutes_daily": random.choice([0, 0, 0, 5, 10, 15, 20]),
    }

def build_user(idx, name, condition):
    age = random.randint(22, 45)
    occupation = random.choice(OCCUPATIONS)
    mood = generate_mood_history(condition)
    stress = generate_stress_scores(condition)
    sleep = generate_sleep_hours(condition)
    energy = generate_energy_levels(condition)
    journal = generate_journal_entries(condition)
    social = generate_social_indicators(condition)

    avg_mood = round(sum(d["score"] for d in mood) / len(mood), 2)
    avg_stress = round(sum(d["score"] for d in stress) / len(stress), 2)
    avg_sleep = round(sum(d["hours"] for d in sleep) / len(sleep), 2)
    avg_energy = round(sum(d["level"] for d in energy) / len(energy), 2)

    return {
        "user_id": f"VT{str(idx+1).zfill(3)}",
        "name": name,
        "age": age,
        "occupation": occupation,
        "condition_label": condition,
        "enrollment_date": (datetime.now() - timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d"),
        "demographics": {
            "gender": random.choice(["M", "F", "NB"]),
            "timezone": random.choice(["UTC+1", "UTC-5", "UTC+5:30", "UTC+8", "UTC-8"]),
        },
        "mood_history": mood,
        "stress_scores": stress,
        "sleep_hours": sleep,
        "energy_levels": energy,
        "journal_entries": journal,
        "social_indicators": social,
        "aggregates": {
            "avg_mood_14d": avg_mood,
            "avg_stress_14d": avg_stress,
            "avg_sleep_14d": avg_sleep,
            "avg_energy_14d": avg_energy,
        }
    }

def generate():
    conditions_dist = (
        ["healthy"] * 8 +
        ["mild_stress"] * 8 +
        ["moderate_stress"] * 8 +
        ["severe_stress"] * 6 +
        ["burnout_risk"] * 7 +
        ["anxiety_trend"] * 6 +
        ["depression_indicators"] * 4 +
        ["resilient"] * 3
    )
    random.shuffle(conditions_dist)

    users = []
    for i, name in enumerate(NAMES):
        condition = conditions_dist[i]
        user = build_user(i, name, condition)
        users.append(user)

    # Save JSON
    os.makedirs("data", exist_ok=True)
    with open("data/users.json", "w") as f:
        json.dump(users, f, indent=2)
    print(f"[✓] Saved data/users.json ({len(users)} users)")

    # Save CSV (flat)
    flat_rows = []
    for u in users:
        row = {
            "user_id": u["user_id"],
            "name": u["name"],
            "age": u["age"],
            "occupation": u["occupation"],
            "condition_label": u["condition_label"],
            "enrollment_date": u["enrollment_date"],
            "avg_mood_14d": u["aggregates"]["avg_mood_14d"],
            "avg_stress_14d": u["aggregates"]["avg_stress_14d"],
            "avg_sleep_14d": u["aggregates"]["avg_sleep_14d"],
            "avg_energy_14d": u["aggregates"]["avg_energy_14d"],
            "social_interactions_per_week": u["social_indicators"]["social_interactions_per_week"],
            "exercise_sessions_per_week": u["social_indicators"]["exercise_sessions_per_week"],
            "screen_time_hours_daily": u["social_indicators"]["screen_time_hours_daily"],
            "work_hours_daily": u["social_indicators"]["work_hours_daily"],
            "latest_journal": u["journal_entries"][-1]["entry"],
        }
        flat_rows.append(row)

    with open("data/users.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_rows[0].keys())
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"[✓] Saved data/users.csv")

    # Save simple SQLite DB
    import sqlite3
    conn = sqlite3.connect("data/vitatwin.db")
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS users")
    c.execute("""
        CREATE TABLE users (
            user_id TEXT PRIMARY KEY,
            name TEXT, age INTEGER, occupation TEXT,
            condition_label TEXT, enrollment_date TEXT,
            avg_mood_14d REAL, avg_stress_14d REAL,
            avg_sleep_14d REAL, avg_energy_14d REAL,
            social_interactions_per_week INTEGER,
            exercise_sessions_per_week INTEGER,
            screen_time_hours_daily REAL,
            work_hours_daily REAL,
            latest_journal TEXT
        )
    """)
    for r in flat_rows:
        c.execute("INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  tuple(r.values()))
    conn.commit()
    conn.close()
    print(f"[✓] Saved data/vitatwin.db (SQLite)")

    return users

if __name__ == "__main__":
    generate()
    print("\n[✓] Dataset generation complete.")
