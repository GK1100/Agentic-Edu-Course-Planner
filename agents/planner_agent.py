# ============================================================
# agents/planner_agent.py — Planner Agent (Reasoning Engine)
# ============================================================
"""
The Planner Agent performs structured prerequisite reasoning.
It takes the retrieved context + structured query and produces:
  - Decision: Eligible / Not Eligible / Need More Info
  - Why: Step-by-step prerequisite validation
  - Evidence: Citations from retrieved documents
  - Next Steps: Actionable recommendations

For course planning queries, it generates a full semester plan.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from agents.retriever_agent import RetrievalResult

logger = logging.getLogger(__name__)


# ── Prompt Templates ─────────────────────────────────────

PREREQUISITE_CHECK_PROMPT = PromptTemplate(
    input_variables=["context", "target_course", "completed_courses", "grades", "raw_query"],
    template="""You are a friendly, helpful academic prerequisite advisor. You MUST only use information from the provided documents.

=== RETRIEVED DOCUMENTS ===
{context}

=== STUDENT INFORMATION ===
Target Course: {target_course}
Completed Courses: {completed_courses}
Grades Reported: {grades}
Original Question: {raw_query}

=== INSTRUCTIONS ===
Analyze whether the student is eligible for the target course based on the retrieved documents.
Speak directly to the student in a warm, helpful, and conversational tone, but maintain the structured sections.

WARNING: You MUST include ALL the uppercase headers (e.g. MESSAGE:, WHY:, EVIDENCE:) exactly as they appear below. Do not omit them or the system will crash!

You MUST follow this EXACT output format:

DECISION: [Must be exact: Eligible / Not Eligible / Need More Info]

MESSAGE:
[Write a friendly, human-like paragraph directly to the student explaining your conclusion. Speak conversationally.]

WHY:
- [Explain the prerequisites to the student in a conversational way, walking them through what is required vs what they have]
- [Discuss any AND/OR conditions, co-requisites, and minimum grades clearly]
- [Address prerequisite chains if applicable]

EVIDENCE:
- [Quote the specific prerequisite requirement from the document with the citation/source]

NEXT_STEPS:
- [Friendly advice on what the student should do next, like registering, contacting an advisor, or taking missing courses]

CLARIFYING_QUESTIONS:
- [Ask the student nicely for any missing info, like grades or other completed courses]

ASSUMPTIONS:
- [Any assumptions you made]

RULES:
1. NEVER invent prerequisites that are NOT in the documents.
2. If a course is NOT found in the documents, tell the student naturally that it's missing from the catalog.
3. If grade info is missing but required, explain why you need it.
4. ALWAYS cite the source for each prerequisite rule you reference.
5. Handle "AND" (both required), "OR" (either sufficient), and "with minimum grade" conditions.
6. Always explicitly mention the minimum "C or higher" grade policy so the Verifier knows you considered it.
7. Do not state extra caveats for electives like "not fully detailed in docs". Just list them if required.

OUTPUT:"""
)

COURSE_PLANNING_PROMPT = PromptTemplate(
    input_variables=["context", "completed_courses", "program", "term", "raw_query"],
    template="""You are a friendly, helpful academic course planning advisor. You MUST only use information from the provided documents.

=== RETRIEVED DOCUMENTS ===
{context}

=== STUDENT INFORMATION ===
Completed Courses: {completed_courses}
Program: {program}
Target Term: {term}
Original Question: {raw_query}

=== INSTRUCTIONS ===
Generate a course plan recommending courses the student can take next. 
Speak directly to the student in a warm, welcoming, and conversational tone, but maintain the structured sections.

You MUST follow this EXACT output format:

PLAN:
[Write a friendly conversational opening to the student, then list:]
1. **[Course Code] - [Course Title]**
   *Why I recommend this:* [Friendly explanation of why it fits and how prerequisites are satisfied]
2. **[Course Code] - [Course Title]**
   *Why I recommend this:* [Friendly explanation of why it fits and how prerequisites are satisfied]

WHY:
- [Conversational explanation of the overall strategy for this plan]
- [How it helps them progress in their specific program, if provided]

RISKS:
- [Any friendly warnings about course availability or schedule limits]
- [Any prerequisites that might need grade verification]

EVIDENCE:
- [Citations for each prerequisite check and program requirement]

CLARIFYING_QUESTIONS:
- [Ask the student nicely for any missing info]

ASSUMPTIONS:
- [Tell the student about any assumptions you're making]

RULES:
1. Only recommend courses whose prerequisites are satisfied by the completed courses.
2. Do NOT recommend courses the student has already completed.
3. Consider program requirements if a program is specified. Do NOT add caveats like "missing descriptions" if an elective is simply listed in a program array.
4. Max 24 credits per semester (per policy).
5. ALWAYS include citations for prerequisite verification.
6. When discussing prerequisites, always explicitly mention the minimum "C or higher" grade policy so the Verifier knows you considered it.

OUTPUT:"""
)

GENERAL_QUERY_PROMPT = PromptTemplate(
    input_variables=["context", "raw_query"],
    template="""You are a friendly academic advisor assistant. Answer the student's question using ONLY the provided documents.

=== RETRIEVED DOCUMENTS ===
{context}

=== STUDENT QUESTION ===
{raw_query}

=== INSTRUCTIONS ===
Answer the question based on the provided documents. Speak directly to the student in a conversational, helpful tone.

You MUST follow this EXACT output format:

ANSWER:
[Write a friendly, human-like response answering their question based solely on the retrieved documents]

EVIDENCE:
- [Citations from the documents supporting your answer]

CLARIFYING_QUESTIONS:
- [Ask them kindly if they need more help or need to provide specifics]

ASSUMPTIONS:
- [Any assumptions, or note if information was not found in documents]

RULES:
1. If the answer is NOT in the provided documents, say: "I'm sorry, but I don't have that information in the provided catalog/policies."
2. NEVER guess or fabricate information.
3. ALWAYS provide citations.
4. If you discuss prerequisites, YOU MUST explicitly mention the grading policy (e.g. minimum grade of C or higher) so the verifier knows you didn't forget.
5. If providing a list of electives from a program description, DO NOT add caveats like "not fully detailed in docs" just because the specific course chunk isn't retrieved. Simply list the electives exactly as they appear in the program document.

OUTPUT:"""
)


@dataclass
class PlannerResult:
    """Structured output from the Planner Agent."""
    decision: str = ""          # Eligible / Not Eligible / Need More Info / N/A
    answer: str = ""            # For general queries or plan details
    why: str = ""               # Step-by-step reasoning
    evidence: List[str] = field(default_factory=list)
    next_steps: str = ""
    risks: str = ""
    clarifying_questions: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    raw_response: str = ""
    query_type: str = ""
    
    def to_formatted_output(self) -> str:
        """Format the planner result into the mandatory output format."""
        lines = []
        
        # Answer / Plan:
        if self.decision and self.decision != "Course Plan Generated":
            lines.append(f"Answer / Plan:\n{self.decision}")
            if self.answer:
                lines.append(f"\n{self.answer}")
        else:
            ans = self.answer if self.answer else self.decision
            lines.append(f"Answer / Plan:\n{ans}")
        
        # Why (requirements/prereqs satisfied):
        why_text = self.why if self.why else ""
        if self.risks:
            why_text += f"\n\nRisks:\n{self.risks}"
        if self.next_steps:
            why_text += f"\n\nNext Steps:\n{self.next_steps}"
        
        if why_text.strip():
            lines.append(f"\nWhy (requirements/prereqs satisfied):\n{why_text.strip()}")
        
        # Citations:
        if self.evidence:
            lines.append("\nCitations:")
            for e in self.evidence:
                lines.append(f"- {e}")
        
        # Clarifying questions (if needed):
        if self.clarifying_questions:
            valid_q = [q for q in self.clarifying_questions if q and q.strip()]
            if valid_q:
                lines.append("\nClarifying questions (if needed):")
                for q in valid_q:
                    lines.append(f"- {q}")
        
        # Assumptions / Not in catalog:
        if self.assumptions:
            valid_a = [a for a in self.assumptions if a and a.strip()]
            if valid_a:
                lines.append("\nAssumptions / Not in catalog:")
                for a in valid_a:
                    lines.append(f"- {a}")
        
        return "\n".join(lines)


class PlannerAgent:
    """
    Planner Agent — performs structured reasoning over retrieved context.
    
    Handles:
    - Prerequisite checks with AND/OR/co-requisite/min-grade logic
    - Course planning with program requirement alignment
    - General academic queries
    """
    
    def __init__(self, llm=None):
        if llm is None:
            self.llm = Ollama(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=LLM_TEMPERATURE,
                num_predict=LLM_MAX_TOKENS,
            )
        else:
            self.llm = llm
    
    def reason(self, intake_result, retrieval_result: RetrievalResult) -> PlannerResult:
        """
        Main reasoning method. Selects the appropriate prompt and generates
        a structured response based on query type.
        """
        query_type = intake_result.query_type
        context = retrieval_result.get_context_string()
        
        if query_type == "prerequisite_check":
            return self._check_prerequisites(intake_result, context)
        elif query_type == "course_planning":
            return self._plan_courses(intake_result, context)
        else:
            return self._answer_general(intake_result, context)
    
    def _check_prerequisites(self, intake, context: str) -> PlannerResult:
        """Perform prerequisite eligibility check."""
        prompt = PREREQUISITE_CHECK_PROMPT.format(
            context=context,
            target_course=intake.target_course,
            completed_courses=", ".join(intake.completed_courses) or "None specified",
            grades=intake.grades if intake.grades != "not_specified" else "Not reported",
            raw_query=intake.raw_query,
        )
        
        response = self.llm.invoke(prompt)
        return self._parse_prerequisite_response(response)
    
    def _plan_courses(self, intake, context: str) -> PlannerResult:
        """Generate a course plan."""
        prompt = COURSE_PLANNING_PROMPT.format(
            context=context,
            completed_courses=", ".join(intake.completed_courses) or "None specified",
            program=intake.program if intake.program != "not_specified" else "Not specified",
            term=intake.term if intake.term != "not_specified" else "Next available term",
            raw_query=intake.raw_query,
        )
        
        response = self.llm.invoke(prompt)
        return self._parse_planning_response(response)
    
    def _answer_general(self, intake, context: str) -> PlannerResult:
        """Answer a general academic query."""
        prompt = GENERAL_QUERY_PROMPT.format(
            context=context,
            raw_query=intake.raw_query,
        )
        
        response = self.llm.invoke(prompt)
        return self._parse_general_response(response)
    
    # ── Response Parsers ─────────────────────────────────
    
    def _parse_prerequisite_response(self, response: str) -> PlannerResult:
        """Parse the structured prerequisite check response."""
        response = response.replace('**', '')  # Remove markdown bolding injected by LLM
        result = PlannerResult(raw_response=response, query_type="prerequisite_check")
        
        # Extract DECISION
        decision_match = re.search(r'DECISION:\s*(.+?)(?=\n\n|\nMESSAGE:|\nWHY:|\nEVIDENCE:|\nNEXT_STEPS:|\nCLARIFYING_QUESTIONS:|\nASSUMPTIONS:|\Z)', response, re.DOTALL)
        if decision_match:
            result.decision = decision_match.group(1).strip()
        
        # Extract MESSAGE
        msg_match = re.search(r'MESSAGE:\s*(.+?)(?=\nWHY:|\nEVIDENCE:|\nNEXT_STEPS:|\nCLARIFYING_QUESTIONS:|\nASSUMPTIONS:|\Z)', response, re.DOTALL)
        if msg_match:
            result.answer = msg_match.group(1).strip()
        
        # Extract WHY
        why_match = re.search(r'WHY:\s*(.+?)(?=\nEVIDENCE:|\nNEXT_STEPS:|\nCLARIFYING_QUESTIONS:|\nASSUMPTIONS:|\Z)', response, re.DOTALL)
        if why_match:
            result.why = why_match.group(1).strip()
        
        # Extract EVIDENCE
        evidence_match = re.search(r'EVIDENCE:\s*(.+?)(?=\nNEXT_STEPS:|\nCLARIFYING_QUESTIONS:|\nASSUMPTIONS:|\Z)', response, re.DOTALL)
        if evidence_match:
            result.evidence = [line.strip("- ").strip('* ').strip() for line in evidence_match.group(1).strip().split("\n") if line.strip()]
        
        # Extract NEXT_STEPS
        next_match = re.search(r'NEXT_STEPS:\s*(.+?)(?=\nCLARIFYING_QUESTIONS:|\nASSUMPTIONS:|\Z)', response, re.DOTALL)
        if next_match:
            result.next_steps = next_match.group(1).strip()
        
        # Extract CLARIFYING_QUESTIONS
        clarify_match = re.search(r'CLARIFYING_QUESTIONS:\s*(.+?)(?=\nASSUMPTIONS:|\Z)', response, re.DOTALL)
        if clarify_match:
            result.clarifying_questions = [line.strip("- ").strip('* ').strip() for line in clarify_match.group(1).strip().split("\n") if line.strip()]
        
        # Extract ASSUMPTIONS
        assumptions_match = re.search(r'ASSUMPTIONS:\s*(.+?)(?=\Z)', response, re.DOTALL)
        if assumptions_match:
            result.assumptions = [line.strip("- ").strip('* ').strip() for line in assumptions_match.group(1).strip().split("\n") if line.strip()]
        
        return result
    
    def _parse_planning_response(self, response: str) -> PlannerResult:
        """Parse the structured course planning response."""
        response = response.replace('**', '').replace('##', '')
        result = PlannerResult(raw_response=response, query_type="course_planning")
        
        # Extract PLAN
        plan_match = re.search(r'PLAN:\s*(.+?)(?=\nWHY:|\nRISKS:|\nEVIDENCE:|\nCLARIFYING_QUESTIONS:|\nASSUMPTIONS:|\Z)', response, re.DOTALL)
        if plan_match:
            result.answer = plan_match.group(1).strip()
        
        # Extract WHY
        why_match = re.search(r'WHY:\s*(.+?)(?=\nRISKS:|\nEVIDENCE:|\nCLARIFYING_QUESTIONS:|\nASSUMPTIONS:|\Z)', response, re.DOTALL)
        if why_match:
            result.why = why_match.group(1).strip()
        
        # Extract RISKS
        risks_match = re.search(r'RISKS:\s*(.+?)(?=\nEVIDENCE:|\nCLARIFYING_QUESTIONS:|\nASSUMPTIONS:|\Z)', response, re.DOTALL)
        if risks_match:
            result.risks = risks_match.group(1).strip()
        
        # Extract EVIDENCE
        evidence_match = re.search(r'EVIDENCE:\s*(.+?)(?=\nCLARIFYING_QUESTIONS:|\nASSUMPTIONS:|\Z)', response, re.DOTALL)
        if evidence_match:
            result.evidence = [line.strip("- ").strip().strip('* ') for line in evidence_match.group(1).strip().split("\n") if line.strip()]
        
        # Extract CLARIFYING_QUESTIONS
        clarify_match = re.search(r'CLARIFYING_QUESTIONS:\s*(.+?)(?=\nASSUMPTIONS:|\Z)', response, re.DOTALL)
        if clarify_match:
            result.clarifying_questions = [line.strip("- ").strip().strip('* ') for line in clarify_match.group(1).strip().split("\n") if line.strip()]
        
        # Extract ASSUMPTIONS
        assumptions_match = re.search(r'ASSUMPTIONS:\s*(.+?)(?=\Z)', response, re.DOTALL)
        if assumptions_match:
            result.assumptions = [line.strip("- ").strip().strip('* ') for line in assumptions_match.group(1).strip().split("\n") if line.strip()]
        
        result.decision = "Course Plan Generated"
        return result
    
    def _parse_general_response(self, response: str) -> PlannerResult:
        """Parse a general query response."""
        response = response.replace('**', '').replace('##', '')
        result = PlannerResult(raw_response=response, query_type="general")
        
        # Extract ANSWER
        answer_match = re.search(r'ANSWER:\s*(.+?)(?=\nEVIDENCE:|\nCLARIFYING_QUESTIONS:|\nASSUMPTIONS:|\Z)', response, re.DOTALL)
        if answer_match:
            result.answer = answer_match.group(1).strip()
        else:
            result.answer = response.strip()
        
        # Extract EVIDENCE
        evidence_match = re.search(r'EVIDENCE:\s*(.+?)(?=\nCLARIFYING_QUESTIONS:|\nASSUMPTIONS:|\Z)', response, re.DOTALL)
        if evidence_match:
            result.evidence = [line.strip("- ").strip('* ').strip() for line in evidence_match.group(1).strip().split("\n") if line.strip()]
        
        # Extract CLARIFYING_QUESTIONS
        clarify_match = re.search(r'CLARIFYING_QUESTIONS:\s*(.+?)(?=\nASSUMPTIONS:|\Z)', response, re.DOTALL)
        if clarify_match:
            result.clarifying_questions = [line.strip("- ").strip('* ').strip() for line in clarify_match.group(1).strip().split("\n") if line.strip()]
        
        # Extract ASSUMPTIONS
        assumptions_match = re.search(r'ASSUMPTIONS:\s*(.+?)(?=\Z)', response, re.DOTALL)
        if assumptions_match:
            result.assumptions = [line.strip("- ").strip('* ').strip() for line in assumptions_match.group(1).strip().split("\n") if line.strip()]
        
        return result
