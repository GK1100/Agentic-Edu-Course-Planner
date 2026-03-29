# ============================================================
# ingestion.py — Document Ingestion & FAISS Indexing Pipeline
# ============================================================
"""
Pipeline: JSON Data → Text Documents → Chunking → Embeddings → FAISS Index

This module handles:
1. Loading course, program, and policy JSON data
2. Converting to LangChain Document objects with metadata
3. Splitting into overlapping chunks for embedding
4. Generating embeddings and storing in a FAISS vector store
"""

import json
import os
import glob
import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    COURSES_DIR, POLICIES_FILE, PROGRAMS_FILE, SOURCES_FILE,
    FAISS_INDEX_DIR, EMBEDDING_MODEL_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── 1. Data Loaders ──────────────────────────────────────

def load_courses() -> List[Document]:
    """Load all course JSON files and convert to Documents."""
    documents = []
    course_files = sorted(glob.glob(os.path.join(COURSES_DIR, "*.json")))
    
    for filepath in course_files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Build a rich text representation of the course
        text = (
            f"Course: {data['course']}\n"
            f"Title: {data['title']}\n"
            f"Credits: {data.get('credits', 'N/A')}\n"
            f"Terms Offered: {', '.join(data.get('terms', []))}\n"
            f"Prerequisites: {data['prerequisites']}\n"
            f"Co-requisites: {data.get('corequisites', 'None')}\n"
            f"Description: {data['description']}\n"
            f"Topics: {', '.join(data.get('topics', []))}\n"
            f"Outcomes: {', '.join(data.get('outcomes', []))}\n"
        )
        
        doc = Document(
            page_content=text,
            metadata={
                "source": f"courses/{os.path.basename(filepath)}",
                "doc_type": "course",
                "course_id": data["course"],
                "course_title": data["title"],
                "prerequisites": data["prerequisites"],
            }
        )
        documents.append(doc)
    
    logger.info(f"Loaded {len(documents)} course documents")
    return documents


def load_programs() -> List[Document]:
    """Load program requirements and convert to Documents."""
    with open(PROGRAMS_FILE, "r", encoding="utf-8") as f:
        programs = json.load(f)
    
    documents = []
    for program in programs:
        parts = [f"Program: {program['program']}"]
        
        if "core_courses" in program:
            parts.append(f"Core Courses: {', '.join(program['core_courses'])}")
        if "advanced_courses" in program:
            parts.append(f"Advanced Courses: {', '.join(program['advanced_courses'])}")
        if "math_requirements" in program:
            parts.append(f"Math Requirements: {', '.join(program['math_requirements'])}")
        if "requirements" in program:
            parts.append(f"Requirements: {', '.join(program['requirements'])}")
        if "electives" in program:
            parts.append(f"Electives: {', '.join(program['electives'])}")
        if "electives_required" in program:
            parts.append(f"Electives Required: {program['electives_required']}")
        if "credits_required" in program:
            parts.append(f"Credits Required: {program['credits_required']}")
        if "residency" in program:
            parts.append(f"Residency Credits Required: {program['residency']}")
        
        text = "\n".join(parts)
        doc = Document(
            page_content=text,
            metadata={
                "source": "programs/programs.json",
                "doc_type": "program",
                "program_name": program["program"],
            }
        )
        documents.append(doc)
    
    logger.info(f"Loaded {len(documents)} program documents")
    return documents


def load_policies() -> List[Document]:
    """Load academic policies and convert to Documents."""
    with open(POLICIES_FILE, "r", encoding="utf-8") as f:
        policies = json.load(f)
    
    documents = []
    for policy in policies:
        text = f"Policy Type: {policy['type']}\nPolicy: {policy['content']}"
        doc = Document(
            page_content=text,
            metadata={
                "source": "policies/policies.json",
                "doc_type": "policy",
                "policy_type": policy["type"],
            }
        )
        documents.append(doc)
    
    logger.info(f"Loaded {len(documents)} policy documents")
    return documents


def load_sources() -> List[Document]:
    """Load sources documentation and convert to Documents."""
    try:
        with open(SOURCES_FILE, "r", encoding="utf-8") as f:
            sources = json.load(f)
    except FileNotFoundError:
        return []
    
    documents = []
    for source in sources:
        text = f"Source Document Info\nURL: {source['url']}\nDate Accessed: {source['date_accessed']}\nDescription: {source['description']}"
        doc = Document(
            page_content=text,
            metadata={
                "source": "sources.json",
                "doc_type": "source",
                "url": source["url"],
            }
        )
        documents.append(doc)
    
    logger.info(f"Loaded {len(documents)} source documents")
    return documents


# ── 2. Chunking ──────────────────────────────────────────

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks.
    
    Chunk size: 500 characters
      - Each course doc is ~1000 chars → ~2 chunks per course
      - This preserves prerequisite info alongside description context
    
    Overlap: 100 characters
      - Ensures prerequisite references at chunk boundaries aren't lost
      - Prevents key info from being split across chunks without context
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    chunks = splitter.split_documents(documents)
    
    # Add chunk index to metadata for citation tracking
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    
    logger.info(
        f"Split {len(documents)} documents into {len(chunks)} chunks "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )
    return chunks


# ── 3. Embedding + FAISS Indexing ────────────────────────

def create_embeddings():
    """Initialize HuggingFace embedding model."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_faiss_index(chunks: List[Document], embeddings) -> FAISS:
    """Build FAISS index from document chunks."""
    logger.info(f"Building FAISS index from {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Persist to disk
    vectorstore.save_local(FAISS_INDEX_DIR)
    logger.info(f"FAISS index saved to {FAISS_INDEX_DIR}")
    
    return vectorstore


def load_faiss_index(embeddings) -> FAISS:
    """Load existing FAISS index from disk."""
    logger.info(f"Loading FAISS index from {FAISS_INDEX_DIR}")
    return FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ── 4. Full Pipeline ────────────────────────────────────

def run_ingestion_pipeline() -> FAISS:
    """
    Complete ingestion pipeline:
    1. Load all data sources (courses, programs, policies)
    2. Chunk documents with overlap
    3. Generate embeddings
    4. Build and persist FAISS index
    
    Returns the FAISS vectorstore ready for retrieval.
    """
    logger.info("=" * 60)
    logger.info("Starting ingestion pipeline...")
    logger.info("=" * 60)
    
    # Step 1: Load all data
    all_docs = []
    all_docs.extend(load_courses())
    all_docs.extend(load_programs())
    all_docs.extend(load_policies())
    all_docs.extend(load_sources())
    logger.info(f"Total documents loaded: {len(all_docs)}")
    
    # Step 2: Chunk
    chunks = chunk_documents(all_docs)
    
    # Step 3: Embed + Index
    embeddings = create_embeddings()
    vectorstore = build_faiss_index(chunks, embeddings)
    
    logger.info("=" * 60)
    logger.info("Ingestion pipeline complete!")
    logger.info("=" * 60)
    
    return vectorstore


if __name__ == "__main__":
    run_ingestion_pipeline()
