# ============================================================
# agents/intake_agent.py — Intake Agent
# ============================================================
"""
The Intake Agent is the entry point of the system.
It parses the user's natural language query and extracts structured fields:
  - completed_courses: courses already taken
  - target_course: the course the user is asking about
  - grades: reported grades
  - program: degree program
  - term: target enrollment term
  - query_type: "prerequisite_check" | "course_planning" | "program_info" | "general"

If critical information is missing, it generates clarifying questions.
"""

import json
import re
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

INTAKE_PROMPT = PromptTemplate(
    input_variables=["user_query"],
    template="""You are an academic intake agent. Analyze the student's query and extract structured information.

STUDENT QUERY: {user_query}

Extract the following fields from the query. If a field is not mentioned, use "not_specified".
Return ONLY a valid JSON object with these exact keys:

{{
  "completed_courses": ["list of courses the student says they have completed"],
  "target_course": "the specific course they are asking about (course code like CS301)",
  "grades": "any grades mentioned (e.g., 'B+ in CS101')",
  "program": "degree program if mentioned (e.g., 'BSc Computer Science')",
  "term": "target term if mentioned (e.g., 'Fall 2025')",
  "query_type": "one of: prerequisite_check, course_planning, program_info, general",
  "clarifying_questions": ["list of questions to ask if critical info is missing"]
}}

RULES:
1. If the student asks about prerequisites for a specific course → query_type = "prerequisite_check"
2. If the student asks what courses to take next → query_type = "course_planning"
3. If the student asks about program requirements → query_type = "program_info"
4. Course codes follow the pattern: CS101, CS201, MATH120, etc.
5. If the student mentions a course name (e.g., "Machine Learning"), map it to the likely code if obvious, otherwise note the name.
6. Only ask clarifying questions if the query type genuinely needs more info.
   - For prerequisite checks: need at minimum the target_course and ideally completed_courses
   - For course planning: need completed_courses and ideally program
7. Return ONLY the JSON, no other text.

JSON Output:"""
)


@dataclass
class IntakeResult:
    """Structured output from the Intake Agent."""
    completed_courses: List[str] = field(default_factory=list)
    target_course: str = "not_specified"
    grades: str = "not_specified"
    program: str = "not_specified"
    term: str = "not_specified"
    query_type: str = "general"
    clarifying_questions: List[str] = field(default_factory=list)
    raw_query: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def has_clarifying_questions(self) -> bool:
        return len(self.clarifying_questions) > 0 and self.clarifying_questions != ["not_specified"]
    
    def needs_more_info(self) -> bool:
        """Check if we have minimum info to proceed."""
        if self.query_type == "prerequisite_check":
            return self.target_course == "not_specified"
        elif self.query_type == "course_planning":
            return len(self.completed_courses) == 0 or self.completed_courses == ["not_specified"]
        return False


class IntakeAgent:
    """
    Parses user queries into structured academic query objects.
    Uses LLM for natural language understanding with fallback regex parsing.
    """
    
    def __init__(self, llm=None):
        if llm is None:
            self.llm = Ollama(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=LLM_TEMPERATURE,
            )
        else:
            self.llm = llm
    
    def process(self, user_query: str) -> IntakeResult:
        """
        Process a user query and return structured intake result.
        Uses LLM parsing with regex fallback.
        """
        logger.info(f"Intake Agent processing: {user_query[:100]}...")
        
        try:
            result = self._llm_parse(user_query)
        except Exception as e:
            logger.warning(f"LLM parsing failed: {e}. Falling back to regex.")
            result = self._regex_parse(user_query)
        
        result.raw_query = user_query
        
        # Post-processing: normalize course codes to uppercase
        result.completed_courses = [c.upper().strip() for c in result.completed_courses 
                                     if c and c != "not_specified"]
        if result.target_course and result.target_course != "not_specified":
            result.target_course = result.target_course.upper().strip()
        
        # Generate clarifying questions if needed
        if result.needs_more_info() and not result.has_clarifying_questions():
            result.clarifying_questions = self._generate_clarifying_questions(result)
        
        logger.info(f"Intake result: type={result.query_type}, target={result.target_course}")
        return result
    
    def _llm_parse(self, user_query: str) -> IntakeResult:
        """Parse using LLM."""
        prompt = INTAKE_PROMPT.format(user_query=user_query)
        response = self.llm.invoke(prompt)
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return IntakeResult(
                completed_courses=data.get("completed_courses", []),
                target_course=data.get("target_course", "not_specified"),
                grades=data.get("grades", "not_specified"),
                program=data.get("program", "not_specified"),
                term=data.get("term", "not_specified"),
                query_type=data.get("query_type", "general"),
                clarifying_questions=data.get("clarifying_questions", []),
            )
        raise ValueError("No valid JSON found in LLM response")
    
    def _regex_parse(self, user_query: str) -> IntakeResult:
        """Fallback regex-based parser for when LLM is unavailable."""
        query_lower = user_query.lower()
        
        # Extract course codes (CS101, MATH220, etc.)
        course_codes = re.findall(r'\b([A-Za-z]{2,4}\d{3})\b', user_query)
        course_codes = [c.upper() for c in course_codes]
        
        # Determine query type
        query_type = "general"
        if any(word in query_lower for word in ["prerequisite", "prereq", "eligible", "can i take", "do i need"]):
            query_type = "prerequisite_check"
        elif any(word in query_lower for word in ["plan", "next semester", "schedule", "what should i take", "recommend"]):
            query_type = "course_planning"
        elif any(word in query_lower for word in ["program", "major", "minor", "degree", "requirement"]):
            query_type = "program_info"
        
        # Determine target course vs completed courses
        target_course = "not_specified"
        completed_courses = []
        
        if query_type == "prerequisite_check" and course_codes:
            # Last mentioned course is likely the target
            target_course = course_codes[-1]
            completed_courses = course_codes[:-1]
        elif query_type == "course_planning":
            completed_courses = course_codes
        elif course_codes:
            target_course = course_codes[0]
            completed_courses = course_codes[1:]
        
        # Extract program names
        program = "not_specified"
        for prog in ["bsc computer science", "ai specialization", "cs minor"]:
            if prog in query_lower:
                program = prog.title()
                break
        
        return IntakeResult(
            completed_courses=completed_courses,
            target_course=target_course,
            query_type=query_type,
            program=program,
        )
    
    def _generate_clarifying_questions(self, result: IntakeResult) -> List[str]:
        """Generate clarifying questions based on what's missing."""
        questions = []
        
        if result.query_type == "prerequisite_check" and result.target_course == "not_specified":
            questions.append("Which course are you checking prerequisites for? Please provide the course code (e.g., CS301).")
        
        if result.query_type == "course_planning" and not result.completed_courses:
            questions.append("Which courses have you already completed? Please list the course codes.")
        
        if result.query_type in ["prerequisite_check", "course_planning"] and not result.completed_courses:
            questions.append("Could you list the courses you have already completed so I can check eligibility?")
        
        if result.grades == "not_specified" and result.query_type == "prerequisite_check":
            questions.append("What grade did you receive in the prerequisite courses? (Some courses require minimum grade C)")
        
        return questions
