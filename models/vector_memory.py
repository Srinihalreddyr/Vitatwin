"""
VitaTwin Vector Memory
Stores conversation turns as TF-IDF FAISS vectors so the assistant
can retrieve semantically relevant past exchanges — giving it genuine
session memory. This is the BONUS vector memory feature.
"""

import os, re, pickle, time
import numpy as np
from collections import Counter
from dataclasses import dataclass, field

try:
    import faiss
    FAISS_OK = True
except ImportError:
    FAISS_OK = False


# ── simple tokeniser (reuses same logic as rag_pipeline) ─────
_STOP = set("the a an is was were are be been have has had do does did will would "
            "could should may might must can i you he she it we they this that of "
            "in on at to for from with by".split())

def _tok(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]+", text.lower())
            if t not in _STOP and len(t) > 2]


@dataclass
class MemoryTurn:
    """One question-answer exchange stored in memory."""
    turn_id:   int
    user_id:   str | None
    question:  str
    answer:    str
    intent:    str
    timestamp: float = field(default_factory=time.time)

    def to_text(self) -> str:
        """Text representation used for vectorisation."""
        return f"{self.question} {self.answer[:300]}"


class VectorMemory:
    """
    FAISS-backed vector memory for the VitaTwin assistant.

    Each conversation turn (question + answer) is embedded using a
    lightweight TF-IDF vectoriser and stored in a FAISS IndexFlatIP.
    On every new question the memory is queried for similar past turns
    so the LLM can reference previous context — true session memory.

    Works with or without FAISS (falls back to cosine-similarity numpy
    search if faiss is unavailable).
    """

    MAX_FEATURES = 256   # vocabulary size for the memory vectoriser

    def __init__(self):
        self._turns:   list[MemoryTurn] = []
        self._vectors: list[np.ndarray] = []
        self._vocab:   list[str] = []
        self._idf:     dict[str, float] = {}
        self._index    = None   # FAISS index, rebuilt after each addition
        self._fitted   = False

    # ── vectorisation ─────────────────────────────────────────

    def _build_vocab(self):
        """Fit TF-IDF vocab on all stored turn texts."""
        docs = [t.to_text() for t in self._turns]
        N = len(docs)
        if N == 0:
            return
        df: Counter = Counter()
        for doc in docs:
            for tok in set(_tok(doc)):
                df[tok] += 1
        top = sorted(df.items(), key=lambda x: -x[1])[:self.MAX_FEATURES]
        self._vocab = [t for t, _ in top]
        self._idf   = {t: float(np.log((N + 1) / (c + 1)) + 1)
                       for t, c in df.items() if t in set(self._vocab)}
        self._fitted = True

    def _vectorise(self, text: str) -> np.ndarray:
        """Convert text to a normalised TF-IDF vector."""
        if not self._fitted:
            return np.zeros(max(self.MAX_FEATURES, 1), dtype=np.float32)
        tokens = _tok(text)
        tf: Counter = Counter(tokens)
        total = max(len(tokens), 1)
        vec = np.zeros(len(self._vocab), dtype=np.float32)
        for i, word in enumerate(self._vocab):
            if word in tf:
                vec[i] = (tf[word] / total) * self._idf.get(word, 1.0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def _rebuild_index(self):
        """Rebuild FAISS index from all stored vectors."""
        if not self._turns or not self._fitted:
            return
        self._vectors = [self._vectorise(t.to_text()) for t in self._turns]
        dim = len(self._vocab)
        if dim == 0:
            return
        matrix = np.stack(self._vectors).astype(np.float32)
        if FAISS_OK:
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(matrix)
        else:
            self._index = matrix  # numpy fallback

    # ── public API ────────────────────────────────────────────

    def store(self, question: str, answer: str,
              user_id: str = None, intent: str = "general"):
        """Store a new conversation turn in vector memory."""
        turn = MemoryTurn(
            turn_id  = len(self._turns),
            user_id  = user_id,
            question = question,
            answer   = answer,
            intent   = intent,
        )
        self._turns.append(turn)
        self._build_vocab()
        self._rebuild_index()

    def retrieve(self, query: str, top_k: int = 3,
                 user_id: str = None) -> list[dict]:
        """
        Retrieve the most semantically similar past turns to the query.
        Optionally filter by user_id.
        Returns list of dicts with turn data + similarity score.
        """
        if not self._turns or not self._fitted:
            return []

        q_vec = self._vectorise(query).reshape(1, -1).astype(np.float32)

        if FAISS_OK and self._index is not None:
            k = min(top_k * 3, len(self._turns))
            scores, indices = self._index.search(q_vec, k)
            results = [(float(scores[0][i]), int(indices[0][i]))
                       for i in range(k) if indices[0][i] >= 0]
        else:
            # numpy cosine similarity fallback
            if self._index is None or len(self._index) == 0:
                return []
            sims = (self._index @ q_vec.T).flatten()
            top_idx = np.argsort(sims)[::-1][:top_k * 3]
            results = [(float(sims[i]), int(i)) for i in top_idx]

        # Filter by user_id if provided, keep top_k
        out = []
        for score, idx in results:
            if idx >= len(self._turns):
                continue
            turn = self._turns[idx]
            if user_id and turn.user_id and turn.user_id != user_id:
                continue
            out.append({
                "turn_id":   turn.turn_id,
                "question":  turn.question,
                "answer":    turn.answer[:300],
                "intent":    turn.intent,
                "user_id":   turn.user_id,
                "similarity": round(score, 4),
                "ago":       f"{int(time.time() - turn.timestamp)}s ago",
            })
            if len(out) >= top_k:
                break

        return out

    def format_for_llm(self, query: str, user_id: str = None,
                       top_k: int = 2) -> str:
        """
        Return a compact string of relevant past turns to inject into
        the LLM prompt so it can reference previous conversation context.
        """
        hits = self.retrieve(query, top_k=top_k, user_id=user_id)
        if not hits:
            return ""
        lines = ["=== RELEVANT CONVERSATION HISTORY ==="]
        for h in hits:
            lines.append(f"[{h['ago']}] Q: {h['question']}")
            lines.append(f"         A: {h['answer'][:200]}...")
        return "\n".join(lines)

    @property
    def size(self) -> int:
        return len(self._turns)

    def clear(self):
        self.__init__()

    def summary(self) -> dict:
        return {
            "turns_stored": self.size,
            "vocab_size":   len(self._vocab),
            "faiss_backend": FAISS_OK,
            "users": list(set(t.user_id for t in self._turns if t.user_id)),
        }
