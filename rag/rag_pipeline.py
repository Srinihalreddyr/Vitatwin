"""
VitaTwin RAG Pipeline
Uses TF-IDF vectorisation + FAISS — 100% local, zero API keys, zero downloads.
Builds a vector index over user profiles and retrieves relevant context.
"""

import json, os, pickle, math, re, numpy as np
from collections import Counter

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "users.json")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faiss.index")
META_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_meta.pkl")

# ─────────────────────────────────────────────
#  Lightweight TF-IDF vectoriser (no sklearn)
# ─────────────────────────────────────────────

STOPWORDS = set("the a an is was were are be been being have has had do does did "
                "will would could should may might must shall can i you he she it "
                "we they this that these those of in on at to for from with by".split())

def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

class TFIDFVectorizer:
    def __init__(self, max_features: int = 512):
        self.max_features = max_features
        self.vocab: list[str] = []
        self.idf: dict[str, float] = {}

    def fit(self, documents: list[str]) -> "TFIDFVectorizer":
        N = len(documents)
        df: Counter = Counter()
        for doc in documents:
            for tok in set(_tokenize(doc)):
                df[tok] += 1
        # Sort by doc-freq descending, keep top features
        top = sorted(df.items(), key=lambda x: -x[1])[:self.max_features]
        self.vocab = [t for t, _ in top]
        vocab_set = set(self.vocab)
        self.idf = {t: math.log((N + 1) / (df[t] + 1)) + 1.0
                    for t in self.vocab}
        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        vecs = []
        for doc in documents:
            tokens = _tokenize(doc)
            tf: Counter = Counter(tokens)
            total = max(len(tokens), 1)
            vec = np.array([
                (tf[t] / total) * self.idf.get(t, 0.0)
                for t in self.vocab
            ], dtype="float32")
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vecs.append(vec)
        return np.stack(vecs)

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)

# ─────────────────────────────────────────────
#  Document builder
# ─────────────────────────────────────────────

def _build_document(user: dict) -> str:
    agg = user["aggregates"]
    si  = user["social_indicators"]
    journals = " | ".join(e["entry"] for e in user["journal_entries"][-7:])
    mood_vals   = [d["score"] for d in user["mood_history"]]
    stress_vals = [d["score"] for d in user["stress_scores"]]
    sleep_vals  = [d["hours"] for d in user["sleep_hours"]]
    mood_trend   = mood_vals[-1]   - mood_vals[0]
    stress_trend = stress_vals[-1] - stress_vals[0]
    sleep_trend  = sleep_vals[-1]  - sleep_vals[0]

    def t(v, inv=False):
        if inv:
            return "decreasing" if v > 0.5 else "increasing" if v < -0.5 else "stable"
        return "increasing" if v > 0.5 else "decreasing" if v < -0.5 else "stable"

    return (
        f"User ID {user['user_id']} name {user['name']} "
        f"age {user['age']} occupation {user['occupation']} "
        f"condition {user['condition_label'].replace('_',' ')} "
        f"average mood {agg['avg_mood_14d']} out of ten trend {t(mood_trend)} "
        f"average stress {agg['avg_stress_14d']} out of ten trend {t(stress_trend)} "
        f"average sleep {agg['avg_sleep_14d']} hours trend {t(sleep_trend, inv=True)} "
        f"average energy {agg['avg_energy_14d']} out of ten "
        f"social interactions {si['social_interactions_per_week']} per week "
        f"exercise {si['exercise_sessions_per_week']} per week "
        f"work hours {si['work_hours_daily']} daily "
        f"screen time {si['screen_time_hours_daily']} hours daily "
        f"journal {journals}"
    )

# ─────────────────────────────────────────────
#  Index builder
# ─────────────────────────────────────────────

def build_index(users: list = None) -> None:
    import faiss
    if users is None:
        with open(DATA_PATH) as f:
            users = json.load(f)

    documents = [_build_document(u) for u in users]
    print(f"[RAG] Vectorising {len(documents)} documents (TF-IDF)…")

    vec = TFIDFVectorizer(max_features=512)
    embeddings = vec.fit_transform(documents)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner-product on unit vectors = cosine
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump({"users": users, "documents": documents, "vectorizer": vec}, f)

    print(f"[RAG] FAISS index built — {index.ntotal} vectors, dim={dim}")

# ─────────────────────────────────────────────
#  Retrieval
# ─────────────────────────────────────────────

_index_cache = None
_meta_cache  = None

def _load_index():
    global _index_cache, _meta_cache
    if _index_cache is None:
        import faiss
        if not os.path.exists(INDEX_PATH):
            build_index()
        _index_cache = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            _meta_cache = pickle.load(f)
    return _index_cache, _meta_cache


def retrieve_by_query(query: str, top_k: int = 5) -> list[dict]:
    index, meta = _load_index()
    vec: TFIDFVectorizer = meta["vectorizer"]
    q_emb = vec.transform([query])
    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "user":       meta["users"][idx],
            "document":   meta["documents"][idx],
            "similarity": round(float(score), 4),
        })
    return results


def retrieve_user_context(user_id: str) -> dict | None:
    _, meta = _load_index()
    for i, u in enumerate(meta["users"]):
        if u["user_id"] == user_id:
            return {"user": u, "document": meta["documents"][i]}
    return None


def retrieve_for_question(question: str, user_id: str = None, top_k: int = 5) -> dict:
    semantic = retrieve_by_query(question, top_k=top_k)
    if user_id:
        user_ctx = retrieve_user_context(user_id)
        return {"target_user": user_ctx, "similar_users": semantic, "question": question}
    return {"target_user": None, "similar_users": semantic, "question": question}


if __name__ == "__main__":
    build_index()
    print("\n--- Query: mental health risks ---")
    for r in retrieve_by_query("What mental health risks are visible?", 3):
        u = r["user"]
        print(f"  {u['user_id']} | {u['name']} | {u['condition_label']} | sim={r['similarity']}")
    print("\n--- Query: stress increased ---")
    for r in retrieve_by_query("Has stress increased recently?", 3):
        u = r["user"]
        print(f"  {u['user_id']} | {u['name']} | stress={u['aggregates']['avg_stress_14d']}")
    print("\n--- User context VT001 ---")
    ctx = retrieve_user_context("VT001")
    if ctx:
        print(f"  {ctx['user']['name']}, {ctx['user']['condition_label']}")
