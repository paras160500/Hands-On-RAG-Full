# 🖼️ Multimodal RAG — Audio, Images & Complex Table Parsing

![Architecture Diagram](https://raw.githubusercontent.com/paras160500/Hands-On-RAG-Full/main/Chapter_8/diagram.png)

Part of the [**Hands-On-RAG-Full**](https://github.com/paras160500/Hands-On-RAG-Full) series. Real-world data isn't just plain text. This module covers **Multimodal RAG**, teaching you how to ingest, process, and retrieve information from non-textual sources like audio conversations, complex multi-page tables, and embedded images within PDFs.

Each technique solves a specific data ingestion challenge:

| Data Type | Challenge | Solution Approach |
|---|---|---|
| 🎙️ **Audio** | Unstructured speech & multiple speakers | Diarized transcription + timestamped chunking |
| 📊 **Complex Tables** | Multi-page breaks & noisy formatting | Docling-based parsing + structural stitching |
| 🖼️ **Images** | Visual data (graphs, diagrams) | Vision-LLM summarization + context-aware insertion |
| 📑 **PDFs** | Mixed media & layout preservation | Advanced document conversion (Docling) |

---

## 🗺️ What's Inside

| Notebook | Technique | Tools |
|---|---|---|
| [`audio_transcribe_for_RAG.ipynb`](#1-audio_transcribe_for_ragipynb--audio-rag-pipelines) | Diarization, Utterance Chunking | `deepgram-sdk`, `python-dotenv` |
| [`image_rag_langchain.ipynb`](#2-image_rag_langchainipynb--multimodal-image-rag) | Vision Summarization, MultiVector Retrieval | `docling`, `langchain`, `openai` |
| [`multi-page-tables.ipynb`](#3-multi-page-tablesipynb--stitching-complex-tables) | Table Stitching, Header Deduplication | `docling`, `pandas`, `requests` |
| [`parse-tables.ipynb`](#4-parse-tablesipynb--advanced-table-extraction) | High-fidelity Table Extraction | `docling` |
| [`simple_table_chunking.ipynb`](#5-simple_table_chunkingipynb--markdown-table-splitting) | Markdown Chunking, Multi-Index Tables | `langchain`, `pandas` |

---

## 📦 Installation

```bash
pip install deepgram-sdk docling langchain langchain-openai pandas pillow requests python-dotenv
```

### 🔑 Environment Variables

Create a `.env` file in this folder:

```env
OPEN_AI_API=your_openai_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
```

---

## 🧪 How Each Notebook Works

---

### 1. `audio_transcribe_for_RAG.ipynb` — Audio RAG Pipelines

Processes call center audio into searchable text chunks. It uses **Deepgram's Nova-2** model to distinguish between speakers (diarization) and creates timestamped metadata for every chunk.

#### Transcription & Metadata Chunking

```python
# Diarized transcription with Deepgram
response = deepgram.listen.v1.media.transcribe_url(
    url=AUDIO_URL,
    model="nova-2",
    diarize=True,
    utterances=True
)

# Creating chunks with speaker info and timestamps
chunks.append({
    "content": "Speaker 0: Hi... Speaker 1: I'm good...",
    "metadata": {"start": 0.16, "end": 145.08, "speakers": ["Speaker 0", "Speaker 1"]}
})
```

---

### 2. `image_rag_langchain.ipynb` — Multimodal Image RAG

Extracts images from complex PDFs using **Docling**, generates textual summaries using **GPT-4o-mini (Vision)**, and indexes both the original text and image summaries for hybrid retrieval.

#### Vision-LLM Summarization

```python
def summarize_image(image_path):
    # GPT-4o-mini analyzes diagrams, graphs, and flowcharts
    # and returns a 4-6 sentence detailed summary.
    pass

# Image summary is stored alongside text chunks
chunks.append({
    "text": f"[IMAGE on page 5] This graph shows a 20% increase in...",
    "metadata": {"source": "image", "image_path": "page_5_img_1.png"}
})
```

---

### 3. `multi-page-tables.ipynb` — Stitching Complex Tables

Solves the problem of tables that span across multiple PDF pages. It uses `Docling` to extract fragments and `Pandas` to logically stitch them back together while removing duplicate headers.

| Step | Action |
|---|---|
| **Extraction** | Use Docling to find all table fragments in the PDF. |
| **Deduplication** | Detect and remove "(continued)" headers on subsequent pages. |
| **Stitching** | Concatenate DataFrames with matching column structures. |

---

### 4. `simple_table_chunking.ipynb` — Markdown Table Splitting

Demonstrates how to handle tables when they are converted to Markdown. It shows the risks of naive chunking (which breaks table structure) and how to manage Multi-Index DataFrames.

```python
# Convert DataFrame to Markdown for RAG context
raw_text = df.to_markdown()

# Naive splitting often breaks the table row-by-row
chunks = text_splitter.split_text(raw_text)
```

---

## 🧠 Choosing the Right Multimodal Tool

```
Dealing with complex PDF layouts and embedded images?
    └── image_rag_langchain.ipynb (Docling + Vision LLM)
         ✅ Preserves structure  ✅ High accuracy  ⚠️ Heavier processing

Processing voice notes or call recordings?
    └── audio_transcribe_for_RAG.ipynb (Deepgram)
         ✅ Fast  ✅ Accurate diarization  ⚠️ Needs API key

Working with massive financial reports/tables?
    └── multi-page-tables.ipynb (Docling + Pandas)
         ✅ Handles page breaks  ✅ Structured data  ⚠️ Requires manual stitching logic
```

---

## ⚡ Tech Stack

| Layer | Tool |
|---|---|
| **Audio Processing** | `Deepgram Nova-2` |
| **PDF Parsing** | `IBM Docling` |
| **Vision Analysis** | `OpenAI GPT-4o-mini` |
| **Vector Store** | `ChromaDB` |
| **Data Handling** | `Pandas` |
| **Orchestration** | `LangChain` |

---

## 🔑 Key Learnings

- **Context is King**: For audio, knowing *who* said *what* and *when* is just as important as the text itself.
- **Vision as a Bridge**: Images in documents often contain the most critical data (charts). Summarizing them into text is a powerful way to make them searchable.
- **Stitching > Splitting**: For tables, it's better to reconstruct the whole table first rather than trying to retrieve individual rows.
- **Docling for Layout**: Traditional PDF parsers fail on complex layouts; specialized tools like Docling are essential for preserving document semantics.

---

## 🚀 Future Improvements

- Implement **Native Multimodal Embeddings** (e.g., CLIP or ColPali) to avoid the summarization step.
- Add **Audio Similarity Search** to find similar sounding segments without full transcription.
- Enhance **Table Reasoning** using specialized LLMs like Table-LLM or GPT-4o's native table parsing.

---

Part of the [**Hands-On-RAG-Full**](https://github.com/paras160500/Hands-On-RAG-Full) repository.
