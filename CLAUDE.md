# CLAUDE.md — AI Video Assistant / Semantic–Temporal Graph Enhanced Retrieval

## Project
RAG pipeline for Q&A and summarization over lecture videos. It serves two purposes:
- Placement portfolio project
- M.Tech Major Project (IT898): "Semantic–Temporal Graph Enhanced Retrieval for Video Question Answering". The earlier minor project, IT897, was "AI Video Assistant".

Baseline paper #1: "Intra-Video Temporal-Aware RAG" (Shafiq et al., CMC 2026), https://www.techscience.com/cmc/v88n2/67662/html

## Stack
- LangChain (LCEL); the LLM goes through `init_chat_model`
- LLM: was `mistral-small-2603`. **Must migrate off Mistral** (free usage ended Nov 2026), so keep the LLM swappable through config.
- ChromaDB vector store, HuggingFace embeddings
- Whisper ASR (keep **word/segment-level timestamps**, because the graph needs them)
- Sarvam AI for Hindi/Hinglish (this replaced deep-translator)
- yt-dlp + ffmpeg for ingestion
- Streamlit UI (`app.py`: dark theme, tabbed results, RAG chat in session state)
- fpdf2 for PDF export

## Research contributions (what we're implementing)
1. **Query-adaptive routing**: classify each query and send it to the right branch:
   - content-lookup → semantic retrieval
   - positional/temporal → temporal branch (uses the occurrence graph)
   - attribute/categorical (e.g. "all red fruits") → graph set-intersection
2. **Intra-video occurrence graph** inside the temporal branch, for disambiguating repeated or same-name entities.

Out of scope: visual frame retrieval.

## Locked graph schema (v1.1): one unified typed property graph
Nodes:
- `Occurrence`: one per mention. Stores its own **fine-grained timestamp** from word/segment-level ASR, not the parent chunk's timestamp.
- `CanonicalEntity`: one per deduplicated entity
- `Concept`, `Attribute`, `Chunk`

Edges:
- `INSTANCE_OF`: Occurrence → CanonicalEntity
- `MENTIONED_IN`: Occurrence → Chunk
- `FOLLOWED_BY`: Occurrence → Occurrence (temporal order; between occurrences, NOT canonical entities)
- `CO_OCCURS_WITH`: Occurrence ↔ CanonicalEntity in the same chunk (symmetric/unordered)
- `IS_A`: CanonicalEntity → Concept
- `HAS_PROPERTY`: CanonicalEntity → Attribute

Deferred (do not implement in v1): `RELATED_TO`, `CAUSES`, `PART_OF`.

Resolution rules:
- Attribute queries: set intersection over `IS_A`/`HAS_PROPERTY` traversals (Fruit ∩ Red ∩ Sweet → Apple, Strawberry).
- Repeated entity, multiple candidate occurrences: disambiguate with `CO_OCCURS_WITH` first (e.g. "after CNN was compared to RNN"), and fall back to embedding similarity over the candidate chunks.
- "What came after X": compare Occurrence timestamps directly. Do not rely on chunk adjacency or `CO_OCCURS_WITH` for ordering.

## Current priorities (in order)
1. Build the graph (schema above) and the graph-aware retrieval, then integrate both into the existing RAG pipeline.
2. Build a concrete worked example showing where the graph beats the baseline paper #1 result (needed for the guide meeting).
3. Run chunk size/overlap sweep experiments with a sweep script against a hand-verified eval set.
4. Generate test data (own test set; LectQA-Vid requested from the base-paper authors). RAGAS evaluation comes later.

## Conventions
- Clean, modular code with error handling; type hints and dataclasses/TypedDicts/Pydantic models for every data structure.
- No deprecated packages or methods; use current LangChain APIs (LCEL, `init_chat_model`).
- Never hardcode or commit API keys. Use `.env` plus `python-dotenv`, and keep `.env` in `.gitignore`.
- When modifying existing scripts or the Streamlit app, keep the existing behavior unless told otherwise. Change the minimum needed.
- Propose a brief plan before large changes.
