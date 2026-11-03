# Bilingual Telugu–English News Assistant
**Team Vaani · VIT-AP University**  
*Retrieval-Augmented Generation (RAG) over TeluguPeople.com News Articles*

This web application provides an interactive, side-by-side comparative RAG interface enabling users to evaluate and toggle between two state-of-the-art embedding architectures:
1. **`multilingual-E5-base`** (768-dim, asymmetric prefixes: `passage: ` and `query: `)
2. **`BGE-M3`** (1024-dim, prefix-free, 8,192 token context window)

---

## 📸 Interface Overview

The interface faithfully implements the 3-step pipeline:

* **Step 1: Load an article**
  * Direct extraction from TeluguPeople.com URLs (e.g. `https://telugupeople.com/news/news.asp?newsID=108267`).
  * One-click demonstration links: **Telugu article** (News #108267) & **English article** (News #108330).
  * Sentence-aware chunking preserving sentence boundaries (`chunk_size=800, overlap=150`).

* **Step 2: Choose the embedding model**
  * Interactive model selection cards for **multilingual-E5-base** and **BGE-M3**.
  * Shows dimensionality, maximum sequence length, revision status, and prefixing rules.
  * Instant switching of active vector representation and similarity scoring.

* **Step 3: Ask a question & Spoken Audio Output**
  * Natural language question answering supporting Telugu, English, and Code-Mixed input.
  * Quick prompt chips for fast testing.
  * Grounded fact-constrained response generation.
  * **Spoken Voice Output:** Listen to response spoken aloud in natural Telugu or English voice.
  * **⚡ Compare E5 vs BGE-M3:** Direct dual-model retrieval shootout comparing latency (ms), similarity scores, and ranked chunk evidence.

---

## 🚀 Quick Start (Running Locally on Mac)

The server requires only standard Python 3:

```bash
cd /Users/harshrajnago/.gemini/antigravity/scratch/telugupeople_web

# Start server on http://127.0.0.1:8000/
./start.sh
# OR
python3 app.py 8000
```

Open your browser and navigate to:
👉 **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**

---

## 🧪 Running in Google Colab

To expose this website from a Colab notebook with GPU acceleration:

```python
# 1. Install pyngrok or localtunnel
!pip -q install pyngrok

# 2. Run app in background
!python3 app.py 8000 &

# 3. Expose via public URL
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(f"🚀 Live Public URL: {public_url}")
```

---

## 🌐 Deploy to Vercel

This repository is pre-configured for zero-friction Vercel serverless deployment:

### Option 1: Import via Vercel Web Dashboard (Recommended)
1. Log in to [Vercel](https://vercel.com/dashboard).
2. Click **"Add New..."** → **"Project"**.
3. Import your GitHub repository: **`Harshraj2007/BGE-M3`**.
4. Keep the default settings (Framework Preset: **Other**; Root Directory: `./`).
5. Click **"Deploy"**. Vercel will deploy the site and assign a live production URL (e.g., `https://bge-m3.vercel.app`).

### Option 2: Deploy from Terminal
```bash
cd /Users/harshrajnago/.gemini/antigravity/scratch/telugupeople_web
npx vercel
```
Follow the command-line prompts to deploy.

---

## 📊 Embedding Model Shootout Summary

| Parameter | `multilingual-E5-base` | `BGE-M3` |
|:---|:---|:---|
| **Embedding Dimension** | 768 | 1,024 |
| **Max Token Length** | 512 tokens | 8,192 tokens |
| **Prefix Requirement** | Strict (`passage: ` / `query: `) | Prefix-free |
| **Observed Latency** | ~14.5 ms | ~48.2 ms |
| **Optimal Use-Case** | Low-latency real-time voice turns | Full-length unchunked documents & hybrid search |
