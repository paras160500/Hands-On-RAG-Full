# 🚀 Advanced RAG Techniques: Caching and Redaction ![Architecture Diagram](diagram.png)

Part of the [**Hands-On-RAG-Full**](https://github.com/paras160500/Hands-On-RAG-Full) series, this module delves into sophisticated strategies for enhancing Retrieval-Augmented Generation (RAG) systems. We focus on two critical aspects: **optimizing performance through intelligent caching mechanisms** and **ensuring data privacy via robust redaction techniques**. This chapter provides practical implementations using Jupyter notebooks, demonstrating how to build more efficient and secure RAG applications.

---

## 💡 What's Inside

| Notebook | What It Covers |
|---|---|
| [`caching.ipynb`](#1-cachingipynb--optimizing-rag-with-caching) | Implement exact-match and semantic caching to boost RAG performance and reduce latency. |
| [`redaction.ipynb`](#2-redactionipynb--ensuring-data-privacy-with-entity-aware-redaction) | Apply entity-aware redaction to protect sensitive information in RAG outputs. |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│ RAG System with Enhancements                            │
│ ┌───────────────────────────────────────────────────┐   │
│ │ Query Input                                       │   │
│ └───────────────────┬───────────────────────────────┘   │
│                     │                                   │
│                     ▼                                   │
│ ┌───────────────────────────────────────────────────┐   │
│ │ Caching Layer (Exact-Match & Semantic)            │   │
│ │   - Redis for Exact-Match Caching                 │   │
│ │   - Embedding Similarity for Semantic Caching     │   │
│ └───────────────────┬───────────────────────────────┘   │
│                     │ Cache Hit / Miss                  │
│                     ▼                                   │
│ ┌───────────────────────────────────────────────────┐   │
│ │ Base Retriever (e.g., FAISS Vector Store)         │   │
│ └───────────────────┬───────────────────────────────┘   │
│                     │ Retrieved Documents               │
│                     ▼                                   │
│ ┌───────────────────────────────────────────────────┐   │
│ │ Redaction Layer (Presidio)                        │   │
│ │   - Detect and Anonymize PII (PERSON, PHONE_NUMBER) │   │
│ └───────────────────┬───────────────────────────────┘   │
│                     │ Redacted Context                  │
│                     ▼                                   │
│ ┌───────────────────────────────────────────────────┐   │
│ │ Language Model (LLM)                              │   │
│ └───────────────────┬───────────────────────────────┘   │
│                     │ Generated Response                │
│                     ▼                                   │
│ ┌───────────────────────────────────────────────────┐   │
│ │ Final Output                                      │   │
│ └───────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

To run the notebooks in this chapter, you'll need the following:

```bash
pip install redis langchain-openai langchain-community langchain-text-splitters langchain-core pydantic presidio-analyzer presidio-anonymizer python-dotenv numpy
```

Additionally, ensure you have Docker installed and running for the Redis cache. The `caching.ipynb` notebook includes commands to start a Redis container.

### 🔑 Environment Variables

Create a `.env` file in this folder with your OpenAI API key:

```env
OPEN_AI_API=your_openai_api_key
```

---

## 🧪 How Each Notebook Works

### 1. `caching.ipynb` — Optimizing RAG with Caching

This notebook demonstrates two caching strategies to improve the efficiency of RAG systems:

#### Exact-Match Caching

Utilizes a hash-based lookup with Redis. If an incoming query's hash matches a previously cached query, the stored results are returned instantly, bypassing the computationally intensive vector search. This is ideal for frequently repeated queries.

```python
class CachedRetriever(BaseRetriever):
    # ... (implementation details)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_hash = hashlib.sha256(query.encode('utf-8')).hexdigest()
        cached_result = cache.get(query_hash)
        if cached_result:
            print("Cache Hit! Skipping the vector search")
            # ... (deserialize and return)

        print("Cache Miss, Performing the vector search ... ")
        results = self._base_retriever.invoke(query)
        cache.setex(query_hash, self._cache_ttl, json.dumps(docs_data))
        return results
```

#### Semantic Caching

Extends caching by using embedding similarity. Queries that are semantically similar, even if phrased differently, can trigger a cache hit. This is achieved by comparing the embedding of the new query with stored embeddings of previous queries using cosine similarity.

```python
class SemanticCachedRetriever(BaseRetriever):
    # ... (implementation details)

    def _find_similar_cached(self, query_embedding: np.ndarray) -> Optional[Tuple[List[Document], str, float]]:
        # ... (cosine similarity calculation)
        if best_similarity >= self._similarity_threshold:
            return best_match, best_query, best_similarity
        return None

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_embedding = np.array(self._embeddings.embed_query(query))
        cache_hit = self._find_similar_cached(query_embedding)
        if cache_hit:
            print(f"Semantic Cache hit. Similarity : {similarity}")
            # ... (return cached docs)

        print("Semantic Cache Miss. Perorming vector search...")
        results = self._base_retriever.invoke(query)
        # ... (store new query and results in cache)
        return results
```

### 2. `redaction.ipynb` — Ensuring Data Privacy with Entity-Aware Redaction

This notebook demonstrates how to implement entity-aware redaction to protect sensitive information within the RAG pipeline. It uses the `presidio` library to identify and replace Personally Identifiable Information (PII) like names and phone numbers with generic placeholders or their entity types.

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

def entity_aware_redaction(text):
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    results = analyzer.analyze(text=text, entities=['PERSON', 'PHONE_NUMBER'], language='en')

    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={
            "PERSON": OperatorConfig("replace", {"new_value": ""}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": ""})
        }
    )
    return anonymized_result.text

input_text = "Dr. Paras called 123-456-7890"
output_text = entity_aware_redaction(input_text)
print(output_text) # Expected: "Dr. called "
```

---

## 📁 Sample Documents

| File | Used In | Content |
|---|---|---|
| `sample_docs.txt` | `caching.ipynb` | Contains sample text related to AI, ML, Deep Learning, NLP, and RAG for demonstration of caching. |

---

## ⚡ Tech Stack

| Layer | Tool |
|---|---|
| RAG Framework | Langchain |
| LLM & Embeddings | OpenAI (via `langchain-openai`) |
| Vector Store | FAISS (via `langchain-community`) |
| Exact-Match Caching | Redis |
| Semantic Caching | Custom implementation with NumPy |
| PII Detection & Redaction | Presidio (via `presidio-analyzer`, `presidio-anonymizer`) |
| Environment Management | `python-dotenv` |

---

## 🧠 Key Learnings

- **Caching is crucial for RAG performance:** Both exact-match and semantic caching significantly reduce latency and computational costs by avoiding redundant vector searches and LLM calls.
- **Semantic caching enhances user experience:** By recognizing semantically similar queries, it provides faster responses even when users rephrase their questions, leading to a more fluid interaction.
- **Data privacy is paramount in RAG:** Implementing entity-aware redaction ensures that sensitive information is protected, making RAG systems suitable for applications handling confidential data.
- **Presidio offers robust PII detection:** It provides a flexible framework for identifying and anonymizing various types of PII, which is essential for compliance and security.

---

## 🚀 Future Improvements

- Integrate a more persistent semantic cache solution (e.g., a dedicated vector database for cache embeddings) rather than in-memory storage.
- Explore advanced redaction techniques, such as format-preserving anonymization or token-level redaction within the LLM generation process.
- Implement a cache invalidation strategy based on document updates or time-based policies.
- Extend redaction to cover more entity types and custom PII patterns.

---

## 👨‍💻 Author

Built for learning: Advanced RAG techniques focusing on caching for performance and redaction for privacy. Made by **PARAS PATEL**.
