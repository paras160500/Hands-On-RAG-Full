# ☁️ Production RAG with Vectara — Ingest, Query, and Correct

Part of the [**Hands-On-RAG-Full**](https://github.com/paras160500/Hands-On-RAG-Full) series. This module moves RAG off a local vector store and onto **Vectara** — a managed, cloud-native RAG platform. It covers the full lifecycle: creating a corpus, ingesting documents (both raw file upload and structured JSON indexing), querying with hybrid search and multilingual re-ranking, correcting hallucinations using Vectara's dedicated correction API, and finally wiring everything into a guardrailed enterprise chatbot.

Every notebook talks directly to the **Vectara REST API v2** using plain `requests` — no SDK, so every request is fully visible and inspectable.

---

## 🚀 What's Inside

| Notebook | What It Covers |
|---|---|
| [`vectara_data_ingestion.ipynb`](#1-vectara_data_ingestionipynb--corpus-creation--document-indexing) | Create a corpus, upload a PDF, and index a structured document with nested sections and metadata |
| [`vectara_ingestion_file_upload.ipynb`](#2-vectara_ingestion_file_uploadipynb--file-upload-standalone) | Focused walkthrough of the file-upload endpoint on its own |
| [`vectara-list-docs.ipynb`](#3-vectara-list-docsipynb--list-retrieve--summarize-documents) | List all documents in a corpus, fetch a specific document, and generate an LLM summary via API |
| [`vectara_query.ipynb`](#4-vectara_queryipynb--hybrid-search-reranking--hallucination-correction) | Run a hybrid search query with re-ranking and factual consistency scoring, then correct a hallucinated answer |
| [`final_project.ipynb`](#5-final_projectipynb--guardrailed-enterprise-rag-chatbot) | A complete employee-handbook chatbot with input guardrails, Vectara RAG, and a confidence threshold gate |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Vectara Cloud                       │
│                                                         │
│   Corpus (RAGBOOK / EMPLOYEE)                           │
│   ┌───────────────────────────────────────────────┐     │
│   │  Chunking + Embedding (managed by Vectara)    │     │
│   │  Multilingual Re-ranker (Rerank_Multilingual) │     │
│   │  Generation (vectara-summary-ext-24-05-...)   │     │
│   │  Hallucination Corrector (vhc-large-1.0)      │     │
│   │  Factual Consistency Scorer                   │     │
│   └───────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
         ▲  upload_file / documents (POST)
         │  query (POST)
         │  correct_hallucinations (POST)
         ▼
  Local Python (requests + dotenv)
         │
         ▼
  Input Guardrails (final_project)
  Confidence Threshold Gate (≥ 0.75)
```

---

## 📦 Installation

```bash
pip install requests python-dotenv
```

No vector store library, no embedding model, no LangChain — Vectara handles all of that server-side.

### 🔑 Environment Variables

Create a `.env` file in this folder:

```env
VECTARA_API_KEY=your_vectara_api_key
```

> Get your API key from the [Vectara Console](https://console.vectara.com). All five notebooks read it via `os.getenv("VECTARA_API_KEY")`.

---

## 🧪 How Each Notebook Works

### 1. `vectara_data_ingestion.ipynb` — Corpus Creation & Document Indexing

**Step 1 — Create a corpus** (idempotent: returns 409 if it already exists):

```python
CORPUS_KEY = "RAGBOOK"

create_corpus_payload = {
    "key": CORPUS_KEY,
    "name": "RAG Example",
    "description": "Corpus for Hands-on RAG"
}

response = requests.post(
    "https://api.vectara.io/v2/corpora",
    headers=headers,
    json=create_corpus_payload
)
# 201 → created, 409 → already exists
```

**Step 2 — Upload a raw PDF** (Vectara handles chunking and embedding automatically):

```python
url = f"https://api.vectara.io/v2/corpora/{CORPUS_KEY}/upload_file"
files = [
    ('file', ('pet_policy', open('pet_policy.pdf', 'rb'), 'application/octet-stream'))
]
response = requests.post(url, headers=headers, files=files)
```

**Step 3 — Index a structured document** (JSON with nested sections, titles, and per-section metadata — useful for books, legal documents, or anything with clear hierarchy):

```python
payload = {
    "id": "selected-works-of-shakespeare",
    "type": "structured",
    "title": "William Shakespeare, Greatest Hits",
    "metadata": {"timespan": "1564–1616", "stars": 5, "author": "William Shakespeare"},
    "sections": [
        {
            "title": "King Lear",
            "text": "Synopsis: ...",
            "sections": [
                {"title": "Act I", "text": "KENT: ...", "metadata": {"stage-instructions": "..."}},
                {"title": "Act II", "text": "EDMUND: ..."}
            ]
        }
    ]
}
response = requests.post(
    f"https://api.vectara.io/v2/corpora/{CORPUS_KEY}/documents",
    headers=headers,
    json=payload
)
```

---

### 2. `vectara_ingestion_file_upload.ipynb` — File Upload Standalone

A focused, minimal version of the PDF upload step — useful as a quick reference or copy-paste template when file upload is all you need:

```python
url = f"https://api.vectara.io/v2/corpora/{corpus_key}/upload_file"
files = [
    ('file', ('pet_policy', open('pet_policy.pdf', 'rb'), 'application/octet-stream'))
]
response = requests.request("POST", url, headers={"x-api-key": api_key}, files=files)
print(response.json())
```

---

### 3. `vectara-list-docs.ipynb` — List, Retrieve & Summarize Documents

**List all documents in the corpus:**
```python
response = requests.get(
    f"https://api.vectara.io/v2/corpora/{CORPUS_KEY}/documents",
    headers=headers
)
```

**Fetch a specific document by its ID:**
```python
response = requests.get(
    f"https://api.vectara.io/v2/corpora/{CORPUS_KEY}/documents/selected-works-of-shakespeare",
    headers=headers
)
```

**Ask Vectara to generate a summary of a specific document** (powered by `gpt-4o` on the backend):
```python
response = requests.post(
    f"https://api.vectara.io/v2/corpora/{CORPUS_KEY}/documents/selected-works-of-shakespeare/summarize",
    headers=headers,
    json={"llm_name": "gpt-4o"}
)
print(res["summary"])
```

---

### 4. `vectara_query.ipynb` — Hybrid Search, Reranking & Hallucination Correction

**Full RAG query** with hybrid search (keyword + semantic), multilingual re-ranking, and a factual consistency score on the generated answer:

```python
payload = {
    "query": "Are pets allowed in the office?",
    "search": {
        "lexical_interpolation": 0.025,      # blend of BM25 + semantic
        "limit": 50,
        "context_configuration": {
            "sentences_before": 2,
            "sentences_after": 2
        },
        "reranker": {
            "type": "customer_reranker",
            "reranker_name": "Rerank_Multilingual_v1"
        }
    },
    "generation": {
        "max_used_search_results": 7,
        "response_language": "eng",
        "generation_preset_name": "vectara-summary-ext-24-05-med-omni",
        "enable_factual_consistency_score": True
    }
}

response = requests.post(f"https://api.vectara.io/v2/corpora/{CORPUS_KEY}/query", ...)
print(res["summary"])
print(f"Factual Consistency Score: {res['factual_consistency_score']}")
```

**Hallucination correction** — feed a deliberately wrong answer and Vectara's `vhc-large-1.0` model patches the specific false claims:

```python
hallucinated_response = """
Pets are allowed in the office at Vectara, but with specific guidelines.
Birds are permitted and even encouraged in the workspace, but there are no rules to follow [2].
However, common household pets like cats and snakes are not allowed on the Vectara campuses [7].
"""

payload = {
    "generated_text": hallucinated_response,
    "query": query_str,
    "documents": [{"text": r["text"]} for r in search_results],
    "model_name": "vhc-large-1.0"
}

response = requests.post(
    "https://api.vectara.io/v2/hallucination_correctors/correct_hallucinations",
    headers=headers,
    json=payload
)

print("Original (hallucinated):", hallucinated_response)
print("Corrected:", res["corrected_text"])
# res["corrections"] → list of specific span-level corrections made
```

---

### 5. `final_project.ipynb` — Guardrailed Enterprise RAG Chatbot

A complete, interactive chatbot built on `employee_handbook.pdf`. It combines three production-readiness features in sequence:

**Input guardrails** — block empty, too-long, or keyword-dangerous queries before they reach the API:

```python
BLOCKED_KEYWORDS = ["hack", "password", "bomb", "malware", "exploit", "terrorist"]

def validate_query(query: str):
    if len(query.strip()) == 0:
        return False, "Query cannot be empty."
    if len(query) > 500:
        return False, "Query too long."
    for word in BLOCKED_KEYWORDS:
        if word in query.lower():
            return False, f"Blocked query detected: {word}"
    return True, ""
```

**RAG query with factual consistency scoring:**

```python
CONFIDENCE_THRESHOLD = 0.75

def generate_answer(query):
    result = ask_vectara(query)
    factual_score = result.get("factual_consistency_score", 0)

    if factual_score < CONFIDENCE_THRESHOLD:
        return {
            "answer": "I couldn't find enough reliable information in the uploaded documents to answer this confidently.",
            "score": factual_score
        }

    return {"answer": result["summary"], "score": factual_score}
```

**Interactive loop — the full pipeline in action:**

```python
upload_document("employee_handbook.pdf")

while True:
    query = input("Ask Question (type exit): ")
    if query.lower() == "exit":
        break

    valid, message = validate_query(query)
    if not valid:
        print(f"Guardrail Blocked: {message}")
        continue

    result = generate_answer(query)
    print("Answer:", result["answer"])
    print("Hallucination Score:", result["score"])
```

---

## 📁 Sample Documents

| File | Used In | Content |
|---|---|---|
| `pet_policy.pdf` | `vectara_data_ingestion.ipynb`, `vectara_ingestion_file_upload.ipynb`, `vectara_query.ipynb` | Office pet policy — the basis for the hallucination correction demo |
| `employee_handbook.pdf` | `final_project.ipynb` | Employee handbook — the knowledge base for the enterprise chatbot |

---

## ⚡ Tech Stack

| Layer | Tool |
|---|---|
| Vector store & RAG | Vectara Cloud (REST API v2) |
| Hybrid search | Vectara (BM25 + semantic, `lexical_interpolation` blend) |
| Re-ranking | Vectara `Rerank_Multilingual_v1` |
| Generation | Vectara `vectara-summary-ext-24-05-med-omni` (backed by OpenAI) |
| Document summarization | Vectara summarize endpoint (backed by `gpt-4o`) |
| Hallucination correction | Vectara `vhc-large-1.0` |
| Factual consistency scoring | Vectara built-in (returned with every query) |
| HTTP client | `requests` |
| Env management | `python-dotenv` |

---

## 🧠 Key Learnings

- **Managed RAG platforms like Vectara absorb the operational complexity** of chunking, embedding, indexing, re-ranking, and generation — the entire pipeline is one POST request, letting you focus on the application logic rather than infrastructure.
- **Structured document indexing** (nested sections + metadata) gives the retriever richer signals than a raw PDF dump — title-level and section-level metadata can be used for filtering and scoring.
- **Hybrid search** (`lexical_interpolation` blending BM25 keyword matching with semantic similarity) consistently outperforms pure vector search for queries with specific proper nouns, product names, or exact phrases.
- **The factual consistency score is not just a diagnostic** — `final_project.ipynb` shows it used as a gate: if the score is below 0.75, the chatbot refuses to answer rather than returning a potentially hallucinated response.
- **Hallucination correction is span-level, not summary-level** — `res["corrections"]` shows exactly which phrases were wrong and what they were replaced with, making corrections auditable.
- Input guardrails should live *before* the API call, not after — blocking on dangerous keywords and length limits is cheap, and every unblocked bad query costs latency and API spend.

---

## 🚀 Future Improvements

- Add **metadata filtering** to the query payload (e.g. filter by document ID or date range) for multi-document corpora
- Pipe the hallucination corrector into the final project's answer path for automatic correction when the factual score is low but not zero
- Add a streaming query endpoint for real-time token-by-token responses in the chatbot
- Log each query, score, and answer to a file or database for offline evaluation and drift monitoring

---

## 👨‍💻 Author

Built for learning: Production RAG with Vectara — cloud ingestion, hybrid search, hallucination correction, and guardrailed enterprise chatbots.

Made by Paras Patel