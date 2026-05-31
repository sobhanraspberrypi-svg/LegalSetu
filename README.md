# 📚 LegalSetu Learn

> **Open-source educational explorer for Indian MSME law**
> Built with strict RAG architecture · Claude API · ChromaDB · Streamlit

[![Open Source](https://img.shields.io/badge/Open%20Source-MIT-green.svg)](LICENSE)
[![Educational](https://img.shields.io/badge/Use-Educational%20Only-orange.svg)](#)
[![RAG](https://img.shields.io/badge/Architecture-Strict%20RAG-blue.svg)](#)

---

## 🎯 What Is This?

LegalSetu Learn is an **educational tool** that helps law students and curious learners understand Indian MSME-related law (MSMED Act, CGST Act, Indian Contract Act) through grounded, citation-backed explanations.

### Key Differentiator: Strict RAG Architecture

Unlike most "AI legal assistants" that hallucinate fake case laws (a real and recurring problem — see [Gujarat HC reprimand of GST officer, Nov 2025](https://www.livelaw.in/)), this tool uses **Retrieval-Augmented Generation (RAG)** to force the AI to answer only from a curated knowledge base of verified public statutes.

If the answer isn't in the knowledge base, Claude is instructed to **refuse rather than guess**.

---

## ⚠️ Important: This Is NOT Legal Advice

- ❌ NOT a substitute for an advocate or CA
- ❌ NOT for filing real legal/tax documents
- ❌ NOT for your specific case
- ✅ FOR understanding general legal concepts
- ✅ FOR law students studying commercial law
- ✅ FOR demonstrating production-grade RAG architecture

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│              MANDATORY AGREEMENT GATE               │
│   (5 click-through acknowledgements required)       │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              USER INPUT (RESTRICTED)                │
│  • Topic Explorer (pre-defined queries)             │
│  • Q&A Mode (with grounding refusal)                │
│  • Knowledge Base Browser (transparency)            │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         VECTOR RETRIEVAL (ChromaDB)                 │
│  • Embeds query using sentence transformers         │
│  • Searches knowledge base for top-4 chunks         │
│  • Returns chunks + source citations                │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         STRICT RAG PROMPT + RETRIEVED CONTEXT       │
│  • System prompt: "Answer ONLY from context"        │
│  • If context insufficient: refuse                  │
│  • Never fabricate case laws                        │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              CLAUDE API (Haiku 4.5)                 │
│         Generates grounded response                 │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│        RESPONSE WITH MANDATORY CITATIONS            │
│   • Answer + 📚 Sources Used + Educational Caveat   │
│   • Raw retrieved chunks shown for transparency     │
└─────────────────────────────────────────────────────┘
```

### Why This Architecture?

| Concern | Solution |
|---|---|
| AI hallucinates fake case laws | RAG forces grounding in verified text |
| Users treat output as legal advice | Mandatory 5-point agreement gate |
| Outputs cannot be verified | Every response shows source citations |
| Users ask about personal cases | Strict refusal of personalized advice |
| Open inputs = anything goes | Structured topic selectors + restricted Q&A |

---

## 📂 Project Structure

```
legalsetu_learn/
├── app.py                      # Main Streamlit UI (agreement gate + 3 tabs)
├── rag_engine.py               # Chunking, embedding, retrieval logic
├── prompts.py                  # Strict RAG system prompt + topic queries
├── requirements.txt            # Dependencies
├── knowledge_base/             # ⭐ The source of truth (verified statutes)
│   ├── 01_msmed_act_2006.md
│   ├── 02_cgst_act_notices.md
│   ├── 03_indian_contract_act.md
│   ├── 04_case_studies.md
│   └── 05_msme_odr_procedure.md
├── data/
│   └── chroma_db/              # Auto-generated vector database (gitignored)
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Local Setup (15 minutes)

### Prerequisites
- Python 3.9+
- A Claude API key from [console.anthropic.com](https://console.anthropic.com) (new accounts get ~$5 free credits)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/legalsetu-learn.git
cd legalsetu-learn

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add API key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml — paste your real ANTHROPIC_API_KEY

# 5. Build the vector index (first time only)
python rag_engine.py

# 6. Run the app
streamlit run app.py
```

App opens at `http://localhost:8501`.

On first launch:
1. Accept the 5 acknowledgements
2. If index isn't built, click "Build Index Now" in the sidebar
3. Try the Topic Explorer first to see grounded responses

---

## ☁️ Deploy to Streamlit Cloud (FREE)

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: LegalSetu Learn RAG MVP"

# Create new public repo on github.com (name: legalsetu-learn)
git remote add origin https://github.com/YOUR_USERNAME/legalsetu-learn.git
git branch -M main
git push -u origin main
```

**⚠️ Verify**: `.streamlit/secrets.toml` is in `.gitignore` — never push your API key!

### Step 2: Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub → "New app"
3. Select `legalsetu-learn` repo, branch `main`, main file `app.py`
4. **Add secret** before deploying — Settings → Secrets:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-api03-YOUR-KEY"
   ```
5. Click **Deploy**

### Step 3: Index Setup on First Deploy

On the first user visit:
- The app will detect the index is empty
- Click "Build Index Now" in the sidebar (one-time, ~30 seconds)
- Index persists across user sessions

---

## 🧠 How RAG Works (For Learners)

### Step 1: Document Chunking
Knowledge base documents (e.g., MSMED Act sections) are split into ~800-character chunks by section headers, preserving context.

### Step 2: Embedding
Each chunk is converted to a 384-dimensional vector using sentence-transformers (`all-MiniLM-L6-v2`). Similar meanings → similar vectors.

### Step 3: Vector Storage
ChromaDB stores these vectors locally in `data/chroma_db/`. Persistent and fast.

### Step 4: Retrieval
When a user asks a question:
- The question is embedded into a vector
- ChromaDB finds the 4 most similar chunks (semantic search, not keyword)
- These chunks are returned with source metadata

### Step 5: Grounded Generation
- Claude receives a strict system prompt: *"Answer ONLY from the retrieved chunks. If insufficient, refuse."*
- The retrieved chunks are passed as user message context
- Claude generates a response strictly grounded in the chunks
- Citations are appended

### Why This Beats Plain Generative
- ✅ No hallucinated case laws
- ✅ Outputs are verifiable (sources shown)
- ✅ Knowledge base is updatable without retraining
- ✅ Refusal when uncertain (vs confidently wrong)

---

## 📚 The Knowledge Base

All content in `knowledge_base/` is:
- Sourced from public-domain Indian statutes
- Hand-curated and verified
- Cited with official URLs
- Updatable independently of code

To **add** to the knowledge base:
1. Create a new `.md` file in `knowledge_base/`
2. Follow the structure of existing files (## headings, citation format)
3. Rerun `python rag_engine.py` to rebuild the index
4. Or click "Rebuild Index" in the app sidebar

---

## 💰 Cost Estimates

Using **Claude Haiku 4.5** (default):
- ~$0.005 per query
- $5 free credits ≈ 1000 queries
- For demo / portfolio use, free tier is more than enough

---

## 🛡️ Safety Features Built In

1. **Mandatory agreement gate** — 5 click-through checkboxes before access
2. **Strict RAG prompt** — Claude is instructed to refuse, not guess
3. **Structured topic selectors** — pre-defined queries reduce open-ended misuse
4. **Citation transparency** — every response shows sources
5. **Raw chunk viewer** — users can verify what Claude saw
6. **Knowledge base browser** — full transparency into what data exists
7. **Persistent disclaimers** — visible on every screen
8. **No document upload** — prevents users from feeding personal notices
9. **No drafting of legal documents** — system prompt explicitly forbids this
10. **Educational framing throughout** — never positioned as advisory

---

## 🎓 Why This Is a Great Portfolio Piece

Demonstrates:
- ✅ **Production RAG architecture** — high-demand AI engineering skill
- ✅ **Vector databases** (ChromaDB) — embeddings, semantic search
- ✅ **LLM grounding & safety** — prompt engineering for reliability
- ✅ **Full-stack ML deployment** — Python + Streamlit + cloud hosting
- ✅ **Responsible AI practices** — safety gates, transparency, refusal
- ✅ **Domain knowledge integration** — bridging legal text with retrieval
- ✅ **Open source contribution** — discoverable, forkable, reviewable

---

## 🤝 Contributing

PRs welcome for:
- More verified knowledge base documents
- Better chunking strategies
- Multi-language support (Hindi, Tamil retrieval)
- Improved retrieval evaluation
- Test cases for grounding verification

Open an issue first to discuss major changes.

---

## 📚 Primary Sources

This tool draws from these public-domain sources. **Always verify against these directly**:

- [MSMED Act 2006](https://msme.gov.in/sites/default/files/MSMED2006act.pdf)
- [CGST Act 2017](https://www.cbic.gov.in/resources//htdocs-cbec/gst/CGST-act-updated.pdf)
- [Indian Contract Act 1872](https://legislative.gov.in/sites/default/files/A1872-09.pdf)
- [Negotiable Instruments Act 1881](https://legislative.gov.in)
- [Indian Kanoon — Case Law Database](https://indiankanoon.org)
- [MSME ODR Portal](https://msmeodr.gov.in)
- [GST Portal](https://www.gst.gov.in)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

**Educational use only. Not legal advice.**

---

## 🙏 Built With

- [Claude API](https://anthropic.com) by Anthropic
- [ChromaDB](https://www.trychroma.com) for vector storage
- [Streamlit](https://streamlit.io) for UI
- [Sentence Transformers](https://www.sbert.net) for embeddings

---

**⭐ Star this repo if it helps you learn RAG architecture or Indian commercial law.**
