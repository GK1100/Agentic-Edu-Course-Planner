# ============================================================
# config.py — Central Configuration for Course Planning Assistant
# ============================================================

import os

# ── Paths ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
COURSES_DIR = os.path.join(DATA_DIR, "Courses")
POLICIES_FILE = os.path.join(DATA_DIR, "Policies", "policies.json")
PROGRAMS_FILE = os.path.join(DATA_DIR, "Programs", "programs.json")
SOURCES_FILE = os.path.join(DATA_DIR, "sources.json")
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")

# ── Embedding Model ───────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── Chunking ──────────────────────────────────────────────
# Chunk size of 500 chars balances context richness with retrieval precision.
# Since each course JSON creates ~1000 chars of text, a chunk size of 500
# means each course maps to ~2 chunks, preserving both description and
# prerequisite info per chunk. Overlap of 100 chars ensures prerequisite
# references at chunk boundaries are not lost.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ── Retrieval ─────────────────────────────────────────────
TOP_K = 15  # Halved from 30 since we have exact metadata filtering now (saves ~40% TTFT)
MULTI_QUERY = True

# ── LLM ───────────────────────────────────────────────────
# Using Ollama with Llama model for local inference
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral-large-3:675b-cloud"  # Change to your locally available model
LLM_TEMPERATURE = 0.1     # Low temperature for factual consistency
LLM_MAX_TOKENS = 2048

# ── Verifier ──────────────────────────────────────────────
MAX_VERIFICATION_RETRIES = 3

# ── Performance Optimizations ─────────────────────────────
# FAST_MODE skips the slow LLM portion of the Verifier Agent, 
# relying purely on rule-based (Regex/String) verification. 
# It cuts response times by roughly ~35%.
FAST_MODE = True
