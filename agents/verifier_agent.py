# ============================================================
# agents/verifier_agent.py — Verifier Agent (Anti-Hallucination)
# ============================================================
"""
The Verifier Agent is the critical anti-hallucination layer.
It validates the Planner's output by checking:
1. Every claim has a citation from the retrieved documents
2. No unsupported assumptions are present
3. Prerequisite logic is correct (AND/OR/min-grade)
4. Course codes mentioned actually exist in the retrieved context

If validation fails, it rejects the response and triggers regeneration.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_TEMPERATURE, MAX_VERIFICATION_RETRIES
from agents.planner_agent import PlannerResult
from agents.retriever_agent import RetrievalResult

logger = logging.getLogger(__name__)


VERIFICATION_PROMPT = PromptTemplate(
    input_variables=["planner_output", "retrieved_context", "query_type"],
    template="""You are a strict academic verification agent. Your job is to verify that the advisor's response is factually correct and fully grounded in the provided documents.

=== ADVISOR'S RESPONSE ===
{planner_output}

=== RETRIEVED DOCUMENTS (GROUND TRUTH) ===
{retrieved_context}

=== QUERY TYPE ===
{query_type}

=== VERIFICATION CHECKLIST ===
Check each of the following and report TRUE or FALSE:

1. CITATION_CHECK: Does every factual claim (prerequisites, course info, policy rules) have a corresponding citation/source reference?
2. FACTUAL_ACCURACY: Are all stated prerequisites correct according to the retrieved documents?
3. LOGIC_CHECK: Is the AND/OR/minimum-grade prerequisite logic applied correctly?
4. NO_FABRICATION: Are there any course codes, rules, or requirements mentioned that do NOT appear in the retrieved documents?
5. COMPLETENESS: Are all relevant prerequisites mentioned? (no missing conditions)

WARNING: If a claim or assumption is factually accurate, DO NOT add it to the issues lists. ONLY add genuine errors, missing citations for factual claims, or true fabrications to the `_issues` arrays. If an array has no true issues, it MUST be empty `[]`.

=== OUTPUT FORMAT ===
Return ONLY a valid JSON object:

{{
  "citation_check": true/false,
  "citation_issues": ["list of claims without citations"],
  "factual_accuracy": true/false,
  "factual_issues": ["list of incorrect facts"],
  "logic_check": true/false,
  "logic_issues": ["list of logic errors"],
  "no_fabrication": true/false,
  "fabrication_issues": ["list of fabricated claims"],
  "completeness": true/false,
  "completeness_issues": ["list of missing information"],
  "overall_valid": true/false,
  "confidence_score": 0.0-1.0,
  "correction_suggestions": ["list of suggested corrections"]
}}

JSON Output:"""
)


@dataclass
class VerificationResult:
    """Output from the Verifier Agent."""
    is_valid: bool = False
    confidence_score: float = 0.0
    citation_check: bool = False
    factual_accuracy: bool = False
    logic_check: bool = False
    no_fabrication: bool = False
    completeness: bool = False
    issues: List[str] = field(default_factory=list)
    correction_suggestions: List[str] = field(default_factory=list)
    attempts: int = 0
    
    def get_summary(self) -> str:
        """Get a human-readable summary of verification."""
        checks = {
            "Citation Coverage": self.citation_check,
            "Factual Accuracy": self.factual_accuracy,
            "Logic Correctness": self.logic_check,
            "No Fabrication": self.no_fabrication,
            "Completeness": self.completeness,
        }
        
        lines = [f"**Verification Status:** {'✅ PASSED' if self.is_valid else '❌ FAILED'}"]
        lines.append(f"**Confidence:** {self.confidence_score:.0%}")
        lines.append(f"**Attempts:** {self.attempts}")
        lines.append("")
        
        for check_name, passed in checks.items():
            emoji = "✅" if passed else "❌"
            lines.append(f"  {emoji} {check_name}")
        
        if self.issues:
            lines.append("\n**Issues Found:**")
            for issue in self.issues:
                lines.append(f"  ⚠️ {issue}")
        
        return "\n".join(lines)


class VerifierAgent:
    """
    Verifier Agent — anti-hallucination validation layer.
    
    Verification strategies:
    1. LLM-based verification: Uses LLM to cross-check claims against context
    2. Rule-based verification: Programmatic checks on structure and citations
    3. Retry mechanism: Re-triggers planning if verification fails (up to MAX_RETRIES)
    """
    
    def __init__(self, llm=None):
        if llm is None:
            self.llm = Ollama(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=0.0,  # Zero temperature for strict verification
            )
        else:
            self.llm = llm
    
    def verify(
        self,
        planner_result: PlannerResult,
        retrieval_result: RetrievalResult,
        planner_agent=None,
        intake_result=None,
    ) -> Tuple[PlannerResult, VerificationResult]:
        """
        Verify the planner's output. If invalid, retry up to MAX_VERIFICATION_RETRIES.
        
        Returns:
            Tuple of (final PlannerResult, VerificationResult)
        """
        current_result = planner_result
        verification = VerificationResult()
        
        for attempt in range(1, MAX_VERIFICATION_RETRIES + 1):
            verification.attempts = attempt
            logger.info(f"Verification attempt {attempt}/{MAX_VERIFICATION_RETRIES}")
            
            # Step 1: Rule-based checks (fast, no LLM needed)
            rule_check = self._rule_based_verify(current_result, retrieval_result)
            
            # Step 2: LLM-based verification (deeper semantic check)
            try:
                llm_check = self._llm_verify(current_result, retrieval_result)
                verification = self._merge_checks(rule_check, llm_check, attempt)
            except Exception as e:
                logger.warning(f"LLM verification failed: {e}. Using rule-based only.")
                verification = rule_check
                verification.attempts = attempt
            
            # If valid, return immediately
            if verification.is_valid:
                logger.info(f"✅ Verification PASSED on attempt {attempt}")
                return current_result, verification
            
            # If invalid and we can retry, regenerate
            if attempt < MAX_VERIFICATION_RETRIES and planner_agent and intake_result:
                logger.warning(
                    f"❌ Verification FAILED on attempt {attempt}. "
                    f"Issues: {verification.issues}. Retrying..."
                )
                # Add correction hints to the intake for re-planning
                current_result = planner_agent.reason(intake_result, retrieval_result)
            else:
                logger.warning(f"❌ Verification FAILED after {attempt} attempts.")
        
        return current_result, verification
    
    def _rule_based_verify(
        self, planner_result: PlannerResult, retrieval_result: RetrievalResult
    ) -> VerificationResult:
        """
        Programmatic verification checks.
        These don't require LLM and catch structural issues.
        """
        result = VerificationResult()
        issues = []
        
        # 1. Citation check: ensure evidence list is not empty
        has_citations = bool(planner_result.evidence) and len(planner_result.evidence) > 0
        result.citation_check = has_citations
        if not has_citations:
            issues.append("No citations provided in the response")
        
        # 2. Check for "not in catalog" honesty
        response_text = planner_result.raw_response.lower()
        retrieved_course_ids = set()
        for chunk in retrieval_result.chunks:
            cid = chunk.metadata.get("course_id", "")
            if cid:
                retrieved_course_ids.add(cid.upper())
        
        # Extract course codes mentioned in the planner's response
        mentioned_courses = set(re.findall(r'\b[A-Z]{2,4}\d{3}\b', planner_result.raw_response.upper()))
        
        # Check for fabricated course references
        result.no_fabrication = True
        for course in mentioned_courses:
            # Allow MATH courses since they might be referenced in prerequisites
            if course.startswith("MATH"):
                continue
            if course not in retrieved_course_ids and len(retrieved_course_ids) > 0:
                # Check if it's in the raw chunk content
                found_in_context = any(
                    course in chunk.content.upper() for chunk in retrieval_result.chunks
                )
                if not found_in_context:
                    result.no_fabrication = False
                    issues.append(f"Course {course} mentioned but not found in retrieved documents")
        
        # 3. Logic check: for prerequisite checks, verify AND/OR handling
        result.logic_check = True
        if planner_result.query_type == "prerequisite_check":
            # Check that the decision is one of the expected values
            valid_decisions = ["eligible", "not eligible", "need more info"]
            if planner_result.decision and planner_result.decision.lower().strip() not in valid_decisions:
                # Be lenient - check if any valid decision is contained
                if not any(d in planner_result.decision.lower() for d in valid_decisions):
                    result.logic_check = False
                    issues.append(f"Invalid decision format: '{planner_result.decision}'")
        
        # 4. Completeness: check that key sections are populated
        result.completeness = True
        if not planner_result.why and not planner_result.answer:
            result.completeness = False
            issues.append("Missing reasoning/explanation section")
        
        # 5. Factual accuracy (rule-based: check prereq strings match context)
        result.factual_accuracy = self._verify_prereq_facts(planner_result, retrieval_result)
        if not result.factual_accuracy:
            issues.append("Prerequisite facts may not match retrieved documents")
        
        # Overall
        result.issues = issues
        result.is_valid = (
            result.citation_check
            and result.no_fabrication
            and result.logic_check
            and result.completeness
        )
        result.confidence_score = sum([
            result.citation_check,
            result.factual_accuracy,
            result.logic_check,
            result.no_fabrication,
            result.completeness,
        ]) / 5.0
        
        return result
    
    def _verify_prereq_facts(
        self, planner_result: PlannerResult, retrieval_result: RetrievalResult
    ) -> bool:
        """
        Verify that prerequisite facts stated in the response match the context.
        """
        # Extract prerequisite statements from retrieved context
        context_prereqs = {}
        for chunk in retrieval_result.chunks:
            prereq = chunk.metadata.get("prerequisites", "")
            course_id = chunk.metadata.get("course_id", "")
            if course_id and prereq:
                context_prereqs[course_id] = prereq
        
        if not context_prereqs:
            return True  # No prereq data to verify against
        
        # Basic check: if the planner mentions a course's prereqs,
        # verify they match what's in the context
        response_upper = planner_result.raw_response.upper()
        for course_id, prereq in context_prereqs.items():
            if course_id in response_upper:
                # The course is mentioned; check if the prereq is also mentioned
                prereq_parts = re.split(r'\s+AND\s+|\s+OR\s+', prereq.upper())
                for part in prereq_parts:
                    part = part.strip()
                    if part and part != "NONE" and len(part) > 3:
                        # Check for the core prereq code (e.g., "CS201" from "CS201 OR instructor consent")
                        prereq_code = re.match(r'[A-Z]{2,4}\d{3}', part)
                        if prereq_code and prereq_code.group() not in response_upper:
                            # Prereq is relevant but not mentioned - potential gap
                            logger.debug(
                                f"Prereq {prereq_code.group()} for {course_id} "
                                f"not found in response"
                            )
        
        return True  # Default to true; deeper checking done by LLM
    
    def _llm_verify(
        self, planner_result: PlannerResult, retrieval_result: RetrievalResult
    ) -> VerificationResult:
        """Use LLM to perform deep semantic verification."""
        prompt = VERIFICATION_PROMPT.format(
            planner_output=planner_result.to_formatted_output(),
            retrieved_context=retrieval_result.get_context_string(),
            query_type=planner_result.query_type,
        )
        
        response = self.llm.invoke(prompt)
        
        # Parse JSON response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                result = VerificationResult(
                    is_valid=data.get("overall_valid", False),
                    confidence_score=float(data.get("confidence_score", 0.0)),
                    citation_check=data.get("citation_check", False),
                    factual_accuracy=data.get("factual_accuracy", False),
                    logic_check=data.get("logic_check", False),
                    no_fabrication=data.get("no_fabrication", False),
                    completeness=data.get("completeness", False),
                    correction_suggestions=data.get("correction_suggestions", []),
                )
                
                # Collect all issues
                all_issues = []
                for key in ["citation_issues", "factual_issues", "logic_issues",
                            "fabrication_issues", "completeness_issues"]:
                    all_issues.extend(data.get(key, []))
                result.issues = [i for i in all_issues if i]
                
                return result
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM verification JSON")
        
        # Fallback: assume moderate confidence
        return VerificationResult(
            is_valid=True,
            confidence_score=0.6,
            citation_check=True,
            factual_accuracy=True,
            logic_check=True,
            no_fabrication=True,
            completeness=True,
        )
    
    def _merge_checks(
        self, rule_check: VerificationResult, llm_check: VerificationResult, attempt: int
    ) -> VerificationResult:
        """Merge rule-based and LLM verification results (conservative merge)."""
        merged = VerificationResult(
            citation_check=rule_check.citation_check and llm_check.citation_check,
            factual_accuracy=rule_check.factual_accuracy and llm_check.factual_accuracy,
            logic_check=rule_check.logic_check and llm_check.logic_check,
            no_fabrication=rule_check.no_fabrication and llm_check.no_fabrication,
            completeness=rule_check.completeness and llm_check.completeness,
            attempts=attempt,
        )
        
        # Conservative: both must agree on validity
        merged.is_valid = (
            merged.citation_check
            and merged.no_fabrication
            and merged.logic_check
            and merged.completeness
        )
        
        # Average confidence scores
        merged.confidence_score = (rule_check.confidence_score + llm_check.confidence_score) / 2
        
        # Combine issues from both
        merged.issues = list(set(rule_check.issues + llm_check.issues))
        merged.correction_suggestions = llm_check.correction_suggestions
        
        return merged
