# ============================================================
# agents/retriever_agent.py — Retriever Agent (RAG Core)
# ============================================================
"""
The Retriever Agent handles similarity search over the FAISS index.
Given a structured query from the Intake Agent, it:
1. Builds an optimized search query
2. Performs FAISS similarity search (top-k)
3. Formats retrieved chunks with source citations
4. Returns structured results with provenance tracking
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import TOP_K

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with metadata."""
    content: str
    source: str
    doc_type: str
    chunk_id: int
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def citation_string(self) -> str:
        """Generate a citation reference string."""
        if self.doc_type == "course":
            course_id = self.metadata.get("course_id", "unknown")
            return f"[Source: {self.source}, Course: {course_id}, Chunk: {self.chunk_id}]"
        elif self.doc_type == "program":
            prog = self.metadata.get("program_name", "unknown")
            return f"[Source: {self.source}, Program: {prog}, Chunk: {self.chunk_id}]"
        elif self.doc_type == "policy":
            pol_type = self.metadata.get("policy_type", "unknown")
            return f"[Source: {self.source}, Policy: {pol_type}, Chunk: {self.chunk_id}]"
        return f"[Source: {self.source}, Chunk: {self.chunk_id}]"


@dataclass
class RetrievalResult:
    """Aggregated retrieval output."""
    chunks: List[RetrievedChunk] = field(default_factory=list)
    query_used: str = ""
    total_found: int = 0
    
    def get_context_string(self) -> str:
        """Build context string for the Planner Agent."""
        parts = []
        for i, chunk in enumerate(self.chunks, 1):
            parts.append(
                f"--- Retrieved Document {i} ---\n"
                f"Source: {chunk.source}\n"
                f"Type: {chunk.doc_type}\n"
                f"Citation: {chunk.citation_string()}\n"
                f"Content:\n{chunk.content}\n"
            )
        return "\n".join(parts)
    
    def get_citation_list(self) -> List[str]:
        """Get all citations as a list."""
        return [chunk.citation_string() for chunk in self.chunks]
    
    def get_chunks_by_type(self, doc_type: str) -> List[RetrievedChunk]:
        """Filter chunks by document type."""
        return [c for c in self.chunks if c.doc_type == doc_type]


class RetrieverAgent:
    """
    Retriever Agent — performs semantic search over the FAISS vector store.
    
    Strategies:
    - For prerequisite checks: searches for the target course AND its prerequisites
    - For course planning: searches for all completed courses + program requirements
    - For program info: searches directly for program-related content
    """
    
    def __init__(self, vectorstore: FAISS, top_k: int = TOP_K):
        self.vectorstore = vectorstore
        self.top_k = top_k
    
    def retrieve(self, intake_result) -> RetrievalResult:
        """
        Main retrieval method. Builds optimized queries based on query type.
        """
        query_type = intake_result.query_type
        
        if query_type == "prerequisite_check":
            return self._retrieve_prerequisites(intake_result)
        elif query_type == "course_planning":
            return self._retrieve_for_planning(intake_result)
        elif query_type == "program_info":
            return self._retrieve_program_info(intake_result)
        else:
            return self._retrieve_general(intake_result)
    
    def _retrieve_prerequisites(self, intake) -> RetrievalResult:
        """Retrieve chunks relevant to prerequisite checking."""
        queries = []
        
        # Primary: search for the target course
        if intake.target_course != "not_specified":
            queries.append(f"Course {intake.target_course} prerequisites requirements")
        
        # Secondary: search for completed courses (to validate prereq chains)
        for course in intake.completed_courses[:5]:  # Limit to avoid too many queries
            queries.append(f"Course {course}")
        
        # Always include policy info for grading rules
        queries.append("prerequisite policy grading minimum grade")
        
        return self._multi_query_retrieve(queries, intake.raw_query, intake.target_course)
    
    def _retrieve_for_planning(self, intake) -> RetrievalResult:
        """Retrieve chunks for course planning."""
        queries = []
        
        # Search for program requirements
        if intake.program != "not_specified":
            queries.append(f"{intake.program} program requirements courses")
        
        # Search for courses the student has completed (to find next steps)
        for course in intake.completed_courses[:5]:
            queries.append(f"Course {course} prerequisites")
        
        # Search for courses that could be next
        queries.append("course prerequisites planning next semester")
        queries.append("academic policy credits semester")
        
        return self._multi_query_retrieve(queries, intake.raw_query)
    
    def _retrieve_program_info(self, intake) -> RetrievalResult:
        """Retrieve program-specific information."""
        queries = []
        
        if intake.program != "not_specified":
            queries.append(f"{intake.program} requirements core courses electives")
        
        if intake.target_course != "not_specified":
            queries.append(f"Course {intake.target_course} program requirements")
        
        queries.append("program requirements credits")
        
        return self._multi_query_retrieve(queries, intake.raw_query, intake.target_course)
    
    def _retrieve_general(self, intake) -> RetrievalResult:
        """General retrieval using the raw query."""
        return self._single_query_retrieve(intake.raw_query)
    
    def _single_query_retrieve(self, query: str) -> RetrievalResult:
        """Perform a single FAISS similarity search."""
        results = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
        return self._format_results(results, query)
    
    def _multi_query_retrieve(self, queries: List[str], raw_query: str, target_course: str = None) -> RetrievalResult:
        """
        Perform multiple queries and deduplicate results.
        This ensures comprehensive coverage for complex queries.
        """
        all_results = {}
        
        # 1. Guaranteed target course extraction
        if target_course and target_course != "not_specified":
            try:
                # Use metadata filtering to grab the exact course logic
                exact_results = self.vectorstore.similarity_search_with_score(
                    target_course, k=10, filter={"course_id": target_course}
                )
                for doc, score in exact_results:
                    chunk_id = doc.metadata.get("chunk_id", id(doc))
                    # Artificially boost score so it is always at the top
                    all_results[chunk_id] = (doc, 0.0)
            except Exception as e:
                logger.warning(f"Exact target course filter failed: {e}")

        # 2. General multi-query extraction
        for query in queries:
            try:
                results = self.vectorstore.similarity_search_with_score(
                    query, k=max(3, self.top_k // 2)
                )
                for doc, score in results:
                    chunk_id = doc.metadata.get("chunk_id", id(doc))
                    if chunk_id not in all_results or score < all_results[chunk_id][1]:
                        all_results[chunk_id] = (doc, score)
            except Exception as e:
                logger.warning(f"Query failed: {query} - {e}")
        
        # 3. Raw query fallback
        try:
            raw_results = self.vectorstore.similarity_search_with_score(
                raw_query, k=self.top_k
            )
            for doc, score in raw_results:
                chunk_id = doc.metadata.get("chunk_id", id(doc))
                if chunk_id not in all_results or score < all_results[chunk_id][1]:
                    all_results[chunk_id] = (doc, score)
        except Exception as e:
            logger.warning(f"Raw query failed: {e}")
        
        # Sort by score (lower is better for L2 distance) and take top-k
        sorted_results = sorted(all_results.values(), key=lambda x: x[1])[:self.top_k]
        
        return self._format_results(sorted_results, raw_query)
    
    def _format_results(self, results, query: str) -> RetrievalResult:
        """Convert FAISS results to structured RetrievalResult."""
        chunks = []
        for doc, score in results:
            chunk = RetrievedChunk(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                doc_type=doc.metadata.get("doc_type", "unknown"),
                chunk_id=doc.metadata.get("chunk_id", -1),
                score=float(score),
                metadata=dict(doc.metadata),
            )
            chunks.append(chunk)
        
        return RetrievalResult(
            chunks=chunks,
            query_used=query,
            total_found=len(chunks),
        )
