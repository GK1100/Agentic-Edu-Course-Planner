# ============================================================
# pipeline.py — Main Orchestration Pipeline
# ============================================================
"""
Orchestrates the full Agentic RAG pipeline:
  User Query → Intake → Retriever → Planner → Verifier → Output

This module ties all four agents together and provides a clean
interface for both CLI and Gradio UI usage.
"""

import os
import logging
from typing import Optional, Tuple

from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS

from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_TEMPERATURE,
    FAISS_INDEX_DIR, LLM_MAX_TOKENS,
)
from ingestion import run_ingestion_pipeline, create_embeddings, load_faiss_index
from agents.intake_agent import IntakeAgent, IntakeResult
from agents.retriever_agent import RetrieverAgent, RetrievalResult
from agents.planner_agent import PlannerAgent, PlannerResult
from agents.verifier_agent import VerifierAgent, VerificationResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


class CoursePlanningPipeline:
    """
    End-to-end Course Planning Assistant pipeline.
    
    Flow:
    1. Intake Agent: Parse user query → structured fields
    2. Retriever Agent: FAISS search → relevant chunks
    3. Planner Agent: Reason over chunks → decision + citations
    4. Verifier Agent: Validate → approve or retry
    """
    
    def __init__(self, llm=None, vectorstore: Optional[FAISS] = None):
        """
        Initialize the pipeline with all agents.
        
        Args:
            llm: Optional LangChain LLM. If None, creates Ollama instance.
            vectorstore: Optional pre-loaded FAISS store. If None, loads from disk.
        """
        # Initialize LLM
        if llm is None:
            self.llm = Ollama(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=LLM_TEMPERATURE,
                num_predict=LLM_MAX_TOKENS,
            )
        else:
            self.llm = llm
        
        # Initialize vector store
        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            self.vectorstore = self._load_or_build_index()
        
        # Initialize agents
        self.intake_agent = IntakeAgent(llm=self.llm)
        self.retriever_agent = RetrieverAgent(vectorstore=self.vectorstore)
        self.planner_agent = PlannerAgent(llm=self.llm)
        self.verifier_agent = VerifierAgent(llm=self.llm)
        
        logger.info("Pipeline initialized with all 4 agents.")
    
    def _load_or_build_index(self) -> FAISS:
        """Load existing FAISS index or build a new one."""
        embeddings = create_embeddings()
        
        if os.path.exists(FAISS_INDEX_DIR):
            logger.info("Loading existing FAISS index...")
            return load_faiss_index(embeddings)
        else:
            logger.info("No FAISS index found. Running ingestion pipeline...")
            return run_ingestion_pipeline()
    
    def process_query(self, user_query: str) -> dict:
        """
        Process a user query through the full pipeline.
        
        Args:
            user_query: Natural language query from the user
            
        Returns:
            Dictionary with all pipeline outputs including:
            - intake: structured query analysis
            - retrieval: retrieved chunks
            - planner: reasoned response
            - verification: validation results
            - final_output: formatted answer
        """
        logger.info("=" * 70)
        logger.info(f"Processing query: {user_query}")
        logger.info("=" * 70)
        
        # ── Step 1: Intake ────────────────────────────────
        logger.info("▶ Step 1: Intake Agent")
        intake_result = self.intake_agent.process(user_query)
        
        # If intake needs clarification, return early with questions
        if intake_result.needs_more_info() and intake_result.has_clarifying_questions():
            return {
                "intake": intake_result.to_dict(),
                "final_output": self._format_clarification_response(intake_result),
                "needs_clarification": True,
            }
        
        # ── Step 2: Retrieval ─────────────────────────────
        logger.info("▶ Step 2: Retriever Agent")
        retrieval_result = self.retriever_agent.retrieve(intake_result)
        logger.info(f"  Retrieved {retrieval_result.total_found} chunks")
        
        # ── Step 3: Planning ──────────────────────────────
        logger.info("▶ Step 3: Planner Agent")
        planner_result = self.planner_agent.reason(intake_result, retrieval_result)
        logger.info(f"  Decision: {planner_result.decision}")
        
        # ── Step 4: Verification ──────────────────────────
        logger.info("▶ Step 4: Verifier Agent")
        verified_result, verification = self.verifier_agent.verify(
            planner_result=planner_result,
            retrieval_result=retrieval_result,
            planner_agent=self.planner_agent,
            intake_result=intake_result,
        )
        logger.info(f"  Verification: {'PASSED' if verification.is_valid else 'FAILED'}")
        logger.info(f"  Confidence: {verification.confidence_score:.0%}")
        
        # ── Build Final Output ────────────────────────────
        final_output = self._build_final_output(verified_result, verification)
        
        return {
            "intake": intake_result.to_dict(),
            "retrieval": {
                "total_chunks": retrieval_result.total_found,
                "query_used": retrieval_result.query_used,
                "citations": retrieval_result.get_citation_list(),
            },
            "planner": {
                "decision": verified_result.decision,
                "answer": verified_result.answer,
                "why": verified_result.why,
                "evidence": verified_result.evidence,
                "next_steps": verified_result.next_steps,
                "risks": verified_result.risks,
            },
            "verification": {
                "is_valid": verification.is_valid,
                "confidence": verification.confidence_score,
                "attempts": verification.attempts,
                "summary": verification.get_summary(),
            },
            "final_output": final_output,
            "needs_clarification": False,
        }
    
    def _build_final_output(
        self, planner_result: PlannerResult, verification: VerificationResult
    ) -> str:
        """Build the mandatory output format."""
        output = planner_result.to_formatted_output()
        
        # Add verification summary
        output += "\n\n---\n"
        output += verification.get_summary()
        
        return output
    
    def _format_clarification_response(self, intake_result: IntakeResult) -> str:
        """Format a response requesting clarification."""
        lines = [
            "**I need some additional information to help you:**\n",
        ]
        for q in intake_result.clarifying_questions:
            lines.append(f"- {q}")
        
        lines.append("\nPlease provide the missing information and I'll assist you.")
        return "\n".join(lines)


# ── CLI Interface ────────────────────────────────────────

def main():
    """Interactive CLI for the Course Planning Assistant."""
    print("=" * 70)
    print("  🎓 Prerequisite & Course Planning Assistant")
    print("  Type your question or 'quit' to exit.")
    print("=" * 70)
    
    pipeline = CoursePlanningPipeline()
    
    while True:
        print()
        user_query = input("📝 Your question: ").strip()
        
        if user_query.lower() in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break
        
        if not user_query:
            continue
        
        try:
            result = pipeline.process_query(user_query)
            print("\n" + "─" * 70)
            print(result["final_output"])
            print("─" * 70)
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
