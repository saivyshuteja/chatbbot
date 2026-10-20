# ContextGuard AI

ContextGuard AI is a local Streamlit app for asking questions about a PDF. It extracts text, splits it into overlapping chunks, retrieves relevant passages with vector and keyword search, and uses Groq to answer questions. PDF-grounded answers include page citations. Explicit requests to explain a general concept (for example, “What is a MultiQueryRetriever?”) are answered from general model knowledge instead of being rejected for lack of PDF evidence.

## Requirements

- Python 3.10 or newer
- A Groq API key
- Internet access for Groq and the first download of the embedding model

## Install and run on Windows

Open PowerShell in the project folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file beside `rag_chatbot.py`:

```dotenv
GROQ_API_KEY=your_groq_api_key
# Optional. Defaults to openai/gpt-oss-20b; fallback models are tried if unavailable.
GROQ_MODEL=openai/gpt-oss-20b
# Optional. Defaults to https://api.groq.com
GROQ_BASE_URL=https://api.groq.com
# Optional. Only needed if Hugging Face requires authentication for the embedding model.
HF_TOKEN=your_hugging_face_token
```

The app also accepts `HUGGINGFACE_HUB_TOKEN` and `HUGGINGFACEHUB_API_TOKEN` for the Hugging Face token. Keep `.env` private and never commit real API keys.

Run the app:

```powershell
streamlit run rag_chatbot.py
```

Upload a text-based PDF (up to 20 MB and 300 pages), click **Process PDF**, then ask a question. Scanned image-only PDFs need OCR, which this version does not include. Indexed chunks are stored locally under `contextguard_storage/chroma_db`.

## How retrieval works

1. PyMuPDF extracts text page by page. The text is split into 350-token chunks with 60-token overlap.
2. `sentence-transformers/all-MiniLM-L6-v2` encodes each chunk and the question as normalized vectors.
3. Chroma uses an HNSW approximate-nearest-neighbor index with cosine distance. For vectors `A` and `B`, cosine similarity is `(A · B) / (||A|| ||B||)` and cosine distance is `1 - similarity`. Lower distance means a closer match. Normalization gives each vector unit length, so vector direction drives the comparison.
4. The app also runs BM25 keyword search. It combines the two ranked lists with Reciprocal Rank Fusion (RRF), adding `1 / (60 + rank)` for each retriever. This combines rankings without treating BM25 scores and cosine distances as though they shared a scale.
5. A relevance gate rejects weak matches without keyword support. The model receives retrieved passages as untrusted evidence and must cite the source page. If evidence is insufficient, the app abstains instead of inventing a PDF answer.

Each answer also has an **Answer evaluation** expander. It shows the original question and grounded answer, retrieved passages with page numbers, cosine similarity between the question and each passage, question-to-answer similarity, and the best answer-to-passage similarity. These embedding scores help inspect semantic relevance; they are not a factual entailment judge, so use the cited passage to verify the answer.

### Cosine versus Euclidean distance

Euclidean distance is `sqrt(sum((A[i] - B[i])^2))`; it measures straight-line separation and changes with vector magnitude. Cosine similarity compares direction and ignores magnitude. With unit-normalized vectors, squared Euclidean distance is `2 - 2 × cosine_similarity`, so both metrics give the same neighbor ordering. This app configures Chroma for cosine distance and normalizes embeddings. HNSW is approximate, so its search is optimized for speed and may not exactly match a full brute-force scan.

## Safety and operational behavior

- Blank questions and questions over 2,000 characters are rejected before retrieval.
- Sensitive credential and personal-data requests are blocked.
- PDF text is treated as untrusted reference content, not instructions for the assistant.
- If no relevant context is found, the app says it could not find relevant evidence. General knowledge is used only for explicit explanation or definition requests that are not asking what the PDF says.
- Groq's client is configured for up to two retries. If a rate limit or quota error remains, the UI explains that the user should wait and try again. Authentication and connection errors also have readable messages.
- PDF limits: 20 MB and 300 pages. Only text-based PDFs are supported.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | required | Groq API credential |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Primary Groq model; configured fallbacks are attempted when unavailable |
| `GROQ_BASE_URL` | `https://api.groq.com` | Native Groq API host; do not append `/openai/v1` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence embedding model |
| `HF_TOKEN` | optional | Hugging Face auth token; alternate names above are supported |
# chatbbot
