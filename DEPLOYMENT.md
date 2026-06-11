# 🚀 Deployment Guide — Streamlit Cloud (Free, No API Key)

## What works on the live deployment
- ✅ Population Overview — all charts, KPIs, scatter plots
- ✅ Individual Profile — trends, radar, clinical findings, signals, journals
- ✅ AI Assistant — conversational check-in + clinical Q&A (template engine)
- ✅ RAG Search — FAISS semantic search across all 50 profiles
- ✅ Vector Memory — session recall
- ✅ Early detection — all risk signals and explainability
- 🟡 LLM responses — uses built-in template engine (Ollama not available on cloud)

> To see full Ollama LLM responses, run locally with `ollama pull llama3`

---

## Step 1 — Create a GitHub repository

1. Go to **https://github.com** and sign in (or create a free account)
2. Click the **+** button → **New repository**
3. Name it `vitatwin`
4. Set it to **Public**
5. Click **Create repository**

---

## Step 2 — Upload your code to GitHub

### Option A — Upload via browser (easiest, no Git needed)
1. Open your `vitatwin` folder on your computer
2. On the GitHub repo page, click **uploading an existing file**
3. Drag and drop **all files and folders** from inside `vitatwin/`
4. Scroll down, click **Commit changes**

> ⚠️ Make sure you upload the contents **inside** the `vitatwin` folder,
> not the folder itself. GitHub should show `README.md`, `ui/`, `data/`, etc. at the root.

### Option B — Git command line
```bash
cd vitatwin
git init
git add .
git commit -m "VitaTwin Mental Health Intelligence System"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/vitatwin.git
git push -u origin main
```

---

## Step 3 — Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io** and sign in with your GitHub account
2. Click **Create app**
3. Fill in:
   - **Repository:** `YOUR_USERNAME/vitatwin`
   - **Branch:** `main`
   - **Main file path:** `ui/dashboard.py`
4. Click **Deploy**
5. Wait 2–3 minutes for it to build

Your live URL will be:
```
https://YOUR_USERNAME-vitatwin-ui-dashboard-XXXX.streamlit.app
```

---

## Troubleshooting

**"ModuleNotFoundError: faiss"**
→ Check that `packages.txt` containing `libgomp1` is in your repo root.

**"No module named ollama"**  
→ This is expected and handled automatically — not an error.

**App loads but shows no data**
→ Make sure `data/faiss.index`, `data/faiss_meta.pkl`, and `data/users.json`
  were all uploaded to GitHub. These must be committed to the repo.

**App crashes on startup**
→ Click **Manage app** → **Logs** in Streamlit Cloud to see the error.
