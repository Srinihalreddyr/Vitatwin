#!/usr/bin/env python3
"""
VitaTwin — Main entry point
Generates dataset, builds FAISS index, and starts the API server or dashboard.

Usage:
    python main.py setup        # Generate data + build FAISS index
    python main.py api          # Start REST API on port 5000
    python main.py test         # Run CLI tests
    streamlit run ui/dashboard.py  # Launch dashboard (run separately)
"""

import sys, os

def setup():
    print("=" * 60)
    print(" VitaTwin Setup")
    print("=" * 60)

    print("\n[1/2] Generating dataset (50 users)...")
    sys.path.insert(0, os.path.dirname(__file__))
    from data.generate_dataset import generate
    generate()

    print("\n[2/2] Building FAISS RAG index...")
    from rag.rag_pipeline import build_index
    build_index()

    print("\n✅  Setup complete!")
    print("\nNext steps:")
    print("  Run API:       python main.py api")
    print("  Run dashboard: streamlit run ui/dashboard.py")
    print("  Run tests:     python main.py test")


def run_api():
    import sys; sys.path.insert(0, ".")
    from api.server import app
    print("[API] Starting on http://localhost:5000")
    app.run(debug=False, port=5000)


def run_tests():
    sys.path.insert(0, ".")
    print("\n" + "=" * 60)
    print("VitaTwin — System Test")
    print("=" * 60)

    from rag.rag_pipeline import retrieve_by_query, retrieve_user_context
    from models.clinical_engine import screen_user
    from models.assistant import MentalHealthAssistant
    import json

    with open("data/users.json") as f:
        users = json.load(f)

    print(f"\n✅  Dataset: {len(users)} users loaded")

    # RAG test
    results = retrieve_by_query("burnout stress exhausted", top_k=3)
    print(f"✅  RAG: Query returned {len(results)} results")
    for r in results:
        print(f"      {r['user']['user_id']} | {r['user']['name']} | sim={r['similarity']:.3f}")

    # Clinical engine test
    test_user = next(u for u in users if u["condition_label"] == "burnout_risk")
    result = screen_user(test_user)
    print(f"✅  Clinical Engine: {test_user['name']} → {result.risk_level} ({result.overall_risk_score}/100)")
    for f in result.findings:
        print(f"      [{f.category}] {f.label} — {len(f.signals)} signals")

    # Assistant test
    assistant = MentalHealthAssistant()
    resp = assistant.ask("What mental health risks are visible?", user_id=test_user["user_id"])
    print(f"✅  Assistant: answered intent='{resp['intent']}'")
    print(f"      Answer preview: {resp['answer'][:100]}...")

    # API test
    from api.server import app
    client = app.test_client()
    r = client.post("/mental/risk-score", json={"user_id": test_user["user_id"]})
    d = r.get_json()
    print(f"✅  API /mental/risk-score: {d['risk_score']}/100 | {d['risk_level']}")

    print("\n" + "=" * 60)
    print("All systems operational ✅")
    print("=" * 60)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "setup"

    if cmd == "setup":
        setup()
    elif cmd == "api":
        run_api()
    elif cmd == "test":
        run_tests()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python main.py [setup|api|test]")
