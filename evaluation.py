# ============================================================
# evaluation.py — Evaluation Script with 25 Test Queries
# ============================================================
"""
Evaluation suite for the Course Planning Assistant.
Tests:
- 10 prerequisite checks
- 5 multi-hop prerequisite chains
- 5 program requirement queries
- 5 "not in docs" cases

Metrics:
- Citation coverage (%)
- Eligibility correctness (%)
- Abstention accuracy (%)
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Test Cases ──────────────────────────────────────────

@dataclass
class TestCase:
    """A single evaluation test case."""
    id: int
    category: str
    query: str
    expected_decision: str          # "Eligible", "Not Eligible", "Need More Info", "Abstain", "Info"
    expected_courses_mentioned: List[str] = field(default_factory=list)
    should_abstain: bool = False    # True if the info is NOT in docs
    description: str = ""


TEST_CASES = [
    # ── 10 Prerequisite Checks ──────────────────────────
    TestCase(
        id=1,
        category="prerequisite_check",
        query="Can I take CS301 Algorithms if I have completed CS201 and CS202?",
        expected_decision="Eligible",
        expected_courses_mentioned=["CS301", "CS201", "CS202"],
        description="Simple AND prerequisite check — both prereqs met",
    ),
    TestCase(
        id=2,
        category="prerequisite_check",
        query="Am I eligible for CS302 Database Systems if I have completed CS201?",
        expected_decision="Eligible",
        expected_courses_mentioned=["CS302", "CS201"],
        description="OR prerequisite — CS201 OR instructor consent",
    ),
    TestCase(
        id=3,
        category="prerequisite_check",
        query="Can I take CS102 Programming II if I got a D in CS101?",
        expected_decision="Not Eligible",
        expected_courses_mentioned=["CS102", "CS101"],
        description="Minimum grade C required — D does not meet it",
    ),
    TestCase(
        id=4,
        category="prerequisite_check",
        query="Can I take CS308 Cybersecurity if I have CS304 but not CS201?",
        expected_decision="Eligible",
        expected_courses_mentioned=["CS308", "CS304"],
        description="OR prerequisite — CS201 OR CS304, student has CS304",
    ),
    TestCase(
        id=5,
        category="prerequisite_check",
        query="Can I take CS305 Machine Learning if I have CS301 but not MATH220?",
        expected_decision="Not Eligible",
        expected_courses_mentioned=["CS305", "CS301", "MATH220"],
        description="AND prerequisite — missing MATH220",
    ),
    TestCase(
        id=6,
        category="prerequisite_check",
        query="Can I take CS101 Introduction to Programming without any prerequisites?",
        expected_decision="Eligible",
        expected_courses_mentioned=["CS101"],
        description="No prerequisites — always eligible",
    ),
    TestCase(
        id=7,
        category="prerequisite_check",
        query="Am I eligible for CS303 Operating Systems with only CS201 completed?",
        expected_decision="Not Eligible",
        expected_courses_mentioned=["CS303", "CS201", "CS202"],
        description="AND prerequisite — missing CS202",
    ),
    TestCase(
        id=8,
        category="prerequisite_check",
        query="Can I take CS310 Distributed Systems if I have CS303 and CS304?",
        expected_decision="Eligible",
        expected_courses_mentioned=["CS310", "CS303", "CS304"],
        description="AND prerequisite — both met",
    ),
    TestCase(
        id=9,
        category="prerequisite_check",
        query="Am I eligible for CS322 Deep Learning if I have CS305 and MATH220?",
        expected_decision="Eligible",
        expected_courses_mentioned=["CS322", "CS305", "MATH220"],
        description="AND prerequisite — both met",
    ),
    TestCase(
        id=10,
        category="prerequisite_check",
        query="Can I take CS302 Database Systems without any courses? I have instructor consent.",
        expected_decision="Eligible",
        expected_courses_mentioned=["CS302"],
        description="OR with instructor consent — should be eligible",
    ),

    # ── 5 Multi-hop Prerequisite Chains ──────────────────
    TestCase(
        id=11,
        category="multi_hop",
        query="I want to take CS305 Machine Learning. I have completed CS101, CS102, MATH120, and MATH220. Am I eligible?",
        expected_decision="Not Eligible",
        expected_courses_mentioned=["CS305", "CS301", "CS201"],
        description="Chain: CS305←CS301←CS201←CS102←CS101. Missing CS201, CS202, CS301",
    ),
    TestCase(
        id=12,
        category="multi_hop",
        query="Can I take CS313 NLP? I have CS101, CS102, MATH120, CS201, CS202, CS301, MATH220, and CS305.",
        expected_decision="Eligible",
        expected_courses_mentioned=["CS313", "CS305"],
        description="Chain: CS313←CS305←CS301←CS201. All prereqs in chain satisfied",
    ),
    TestCase(
        id=13,
        category="multi_hop",
        query="I want to take CS320 Advanced AI Systems. I have CS101, CS102, CS201, CS202, and CS301. Am I eligible?",
        expected_decision="Not Eligible",
        expected_courses_mentioned=["CS320", "CS306", "CS301"],
        description="Chain: CS320←CS306←CS301. Missing CS306",
    ),
    TestCase(
        id=14,
        category="multi_hop",
        query="Can I take CS310 Distributed Systems? I have CS101, CS102, MATH120, CS201, CS202, CS303, and CS304.",
        expected_decision="Eligible",
        expected_courses_mentioned=["CS310", "CS303", "CS304"],
        description="Chain: CS310←CS303+CS304, CS303←CS201+CS202. All satisfied",
    ),
    TestCase(
        id=15,
        category="multi_hop",
        query="Can I take CS321 Reinforcement Learning? I have only CS101 and CS102.",
        expected_decision="Not Eligible",
        expected_courses_mentioned=["CS321", "CS305"],
        description="Chain: CS321←CS305←CS301←CS201+CS202. Many missing prereqs",
    ),

    # ── 5 Program Requirement Queries ────────────────────
    TestCase(
        id=16,
        category="program_info",
        query="What are the core courses required for BSc Computer Science?",
        expected_decision="Info",
        expected_courses_mentioned=["CS101", "CS102", "CS201", "CS202", "CS301"],
        description="Program info query — core courses",
    ),
    TestCase(
        id=17,
        category="program_info",
        query="What courses do I need for the AI Specialization?",
        expected_decision="Info",
        expected_courses_mentioned=["CS301", "CS305", "CS306"],
        description="Program info query — AI specialization requirements",
    ),
    TestCase(
        id=18,
        category="program_info",
        query="What are the requirements for the CS Minor?",
        expected_decision="Info",
        expected_courses_mentioned=["CS101", "CS102", "CS201"],
        description="Program info query — CS Minor",
    ),
    TestCase(
        id=19,
        category="program_info",
        query="How many credits are required for BSc Computer Science?",
        expected_decision="Info",
        description="Program info query — credit requirement (120 credits)",
    ),
    TestCase(
        id=20,
        category="program_info",
        query="What math courses are required for BSc Computer Science?",
        expected_decision="Info",
        expected_courses_mentioned=["MATH120", "MATH220", "MATH230"],
        description="Program info query — math requirements",
    ),

    # ── 5 "Not in Docs" Cases ────────────────────────────
    TestCase(
        id=21,
        category="not_in_docs",
        query="What are the prerequisites for CS400 Advanced Robotics?",
        expected_decision="Abstain",
        should_abstain=True,
        description="Course CS400 does NOT exist in the catalog",
    ),
    TestCase(
        id=22,
        category="not_in_docs",
        query="Can I take ENG201 Technical Writing? I have CS101.",
        expected_decision="Abstain",
        should_abstain=True,
        description="ENG201 is NOT in the catalog",
    ),
    TestCase(
        id=23,
        category="not_in_docs",
        query="What is the tuition fee for BSc Computer Science?",
        expected_decision="Abstain",
        should_abstain=True,
        description="Tuition info is NOT in the catalog/policies",
    ),
    TestCase(
        id=24,
        category="not_in_docs",
        query="When does the Spring 2025 semester start?",
        expected_decision="Abstain",
        should_abstain=True,
        description="Academic calendar is NOT in the catalog",
    ),
    TestCase(
        id=25,
        category="not_in_docs",
        query="What are the admission requirements for the Masters in Computer Science?",
        expected_decision="Abstain",
        should_abstain=True,
        description="Masters program is NOT in the catalog",
    ),
]


# ── Scoring Functions ────────────────────────────────────

def score_citation_coverage(result: Dict[str, Any]) -> float:
    """
    Score: What % of claims have citations?
    Returns a value between 0.0 and 1.0.
    """
    planner = result.get("planner", {})
    evidence = planner.get("evidence", [])
    
    if not evidence:
        return 0.0
    
    # Count non-empty citations
    valid_citations = [e for e in evidence if e and e.strip() and len(e) > 5]
    
    # Expect at least 1 citation
    if len(valid_citations) >= 1:
        return min(1.0, len(valid_citations) / max(1, len(evidence)))
    
    return 0.0


def score_eligibility_correctness(result: Dict[str, Any], test_case: TestCase) -> float:
    """
    Score: Is the eligibility decision correct?
    Returns 1.0 for correct, 0.0 for incorrect.
    """
    if test_case.should_abstain:
        return score_abstention(result, test_case)
    
    planner = result.get("planner", {})
    decision = planner.get("decision", "").lower().strip()
    if not decision:
        decision = result.get("final_output", "").lower()
    final_output = result.get("final_output", "").lower()
    
    expected = test_case.expected_decision.lower()
    
    # Check if expected decision appears in the response
    if expected == "info":
        # For info queries, just check that an answer was provided
        answer = planner.get("answer", "")
        if answer and len(answer) > 20:
            return 1.0
        # Check final output
        if len(final_output) > 50:
            return 0.7
        return 0.0
    
    # For eligibility decisions
    if expected in decision:
        return 1.0
    
    # Check if it's in the final output
    if expected in final_output:
        return 0.8
    
    # Partial credit for "need more info" when correct answer is Known
    if "need more info" in decision and expected in ["eligible", "not eligible"]:
        return 0.3
    
    return 0.0


def score_abstention(result: Dict[str, Any], test_case: TestCase) -> float:
    """
    Score: Does the system correctly abstain when info is not in docs?
    Returns 1.0 if it correctly says "not in catalog/docs".
    """
    final_output = result.get("final_output", "").lower()
    planner = result.get("planner", {})
    answer = planner.get("answer", "").lower()
    decision = planner.get("decision", "").lower()
    
    abstention_phrases = [
        "not in the",
        "not found in",
        "don't have that information",
        "not in catalog",
        "not in the provided",
        "not available in",
        "no information",
        "not included",
        "does not exist",
        "not in docs",
        "cannot find",
        "outside the scope",
    ]
    
    combined_text = f"{final_output} {answer} {decision}"
    
    for phrase in abstention_phrases:
        if phrase in combined_text:
            return 1.0
    
    # Partial credit: if it's vague but doesn't fabricate
    if "not eligible" not in combined_text and "eligible" not in combined_text:
        return 0.5
    
    return 0.0


def score_course_mentions(result: Dict[str, Any], test_case: TestCase) -> float:
    """Score: Are the expected courses mentioned in the response?"""
    if not test_case.expected_courses_mentioned:
        return 1.0
    
    final_output = result.get("final_output", "").upper()
    
    mentioned = 0
    for course in test_case.expected_courses_mentioned:
        if course.upper() in final_output:
            mentioned += 1
    
    return mentioned / len(test_case.expected_courses_mentioned)


# ── Evaluation Runner ────────────────────────────────────

@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics."""
    total_tests: int = 0
    
    citation_coverage_scores: List[float] = field(default_factory=list)
    eligibility_scores: List[float] = field(default_factory=list)
    abstention_scores: List[float] = field(default_factory=list)
    course_mention_scores: List[float] = field(default_factory=list)
    
    category_scores: Dict[str, List[float]] = field(default_factory=dict)
    
    individual_results: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def citation_coverage(self) -> float:
        if not self.citation_coverage_scores:
            return 0.0
        return sum(self.citation_coverage_scores) / len(self.citation_coverage_scores)
    
    @property
    def eligibility_correctness(self) -> float:
        if not self.eligibility_scores:
            return 0.0
        return sum(self.eligibility_scores) / len(self.eligibility_scores)
    
    @property
    def abstention_accuracy(self) -> float:
        if not self.abstention_scores:
            return 0.0
        return sum(self.abstention_scores) / len(self.abstention_scores)
    
    def get_report(self) -> str:
        """Generate a formatted evaluation report."""
        lines = [
            "=" * 70,
            "  📊 EVALUATION REPORT",
            "=" * 70,
            "",
            f"  Total Tests:            {self.total_tests}",
            "",
            "  ── Overall Metrics ──",
            f"  Citation Coverage:      {self.citation_coverage:.1%}",
            f"  Eligibility Correctness:{self.eligibility_correctness:.1%}",
            f"  Abstention Accuracy:    {self.abstention_accuracy:.1%}",
            "",
            "  ── By Category ──",
        ]
        
        for category, scores in self.category_scores.items():
            avg = sum(scores) / len(scores) if scores else 0
            lines.append(f"  {category:25s} {avg:.1%}  ({len(scores)} tests)")
        
        lines.append("")
        lines.append("  ── Individual Results ──")
        lines.append(f"  {'ID':>4s} | {'Category':15s} | {'Citation':>8s} | {'Correct':>7s} | {'Status':6s} | Description")
        lines.append("  " + "-" * 85)
        
        for r in self.individual_results:
            status = "✅" if r["eligibility_score"] >= 0.7 else "❌"
            lines.append(
                f"  {r['id']:4d} | {r['category']:15s} | "
                f"{r['citation_score']:>7.0%} | "
                f"{r['eligibility_score']:>6.0%} | "
                f"{status:6s} | {r['description'][:40]}"
            )
        
        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)


def run_evaluation(
    pipeline=None,
    test_cases: Optional[List[TestCase]] = None,
    save_results: bool = True,
) -> EvaluationMetrics:
    """
    Run the full evaluation suite.
    
    Args:
        pipeline: CoursePlanningPipeline instance. If None, creates one.
        test_cases: List of TestCase objects. If None, uses default TEST_CASES.
        save_results: Whether to save results to a JSON file.
    
    Returns:
        EvaluationMetrics with all scores.
    """
    if pipeline is None:
        from pipeline import CoursePlanningPipeline
        pipeline = CoursePlanningPipeline()
    
    if test_cases is None:
        test_cases = TEST_CASES
    
    metrics = EvaluationMetrics()
    metrics.total_tests = len(test_cases)
    
    logger.info(f"Running evaluation with {len(test_cases)} test cases...")
    
    for i, tc in enumerate(test_cases, 1):
        logger.info(f"\n[{i}/{len(test_cases)}] Test {tc.id}: {tc.description}")
        logger.info(f"  Query: {tc.query}")
        
        try:
            start = time.time()
            result = pipeline.process_query(tc.query)
            elapsed = time.time() - start
            
            # Score
            citation_score = score_citation_coverage(result)
            eligibility_score = score_eligibility_correctness(result, tc)
            course_score = score_course_mentions(result, tc)
            
            # Track abstention separately
            if tc.should_abstain:
                abstention_score = score_abstention(result, tc)
                metrics.abstention_scores.append(abstention_score)
            
            metrics.citation_coverage_scores.append(citation_score)
            metrics.eligibility_scores.append(eligibility_score)
            metrics.course_mention_scores.append(course_score)
            
            # Category tracking
            if tc.category not in metrics.category_scores:
                metrics.category_scores[tc.category] = []
            metrics.category_scores[tc.category].append(eligibility_score)
            
            # Individual result
            metrics.individual_results.append({
                "id": tc.id,
                "category": tc.category,
                "query": tc.query,
                "description": tc.description,
                "expected_decision": tc.expected_decision,
                "actual_decision": result.get("planner", {}).get("decision", "N/A"),
                "citation_score": citation_score,
                "eligibility_score": eligibility_score,
                "course_mention_score": course_score,
                "time_seconds": round(elapsed, 2),
                "verification_valid": result.get("verification", {}).get("is_valid", False),
            })
            
            status = "✅" if eligibility_score >= 0.7 else "❌"
            logger.info(
                f"  {status} Citation: {citation_score:.0%}, "
                f"Correct: {eligibility_score:.0%}, "
                f"Time: {elapsed:.1f}s"
            )
            
        except Exception as e:
            logger.error(f"  ❌ Test {tc.id} FAILED: {e}")
            metrics.eligibility_scores.append(0.0)
            metrics.citation_coverage_scores.append(0.0)
            metrics.course_mention_scores.append(0.0)
            
            if tc.category not in metrics.category_scores:
                metrics.category_scores[tc.category] = []
            metrics.category_scores[tc.category].append(0.0)
            
            metrics.individual_results.append({
                "id": tc.id,
                "category": tc.category,
                "query": tc.query,
                "description": tc.description,
                "expected_decision": tc.expected_decision,
                "actual_decision": f"ERROR: {e}",
                "citation_score": 0.0,
                "eligibility_score": 0.0,
                "course_mention_score": 0.0,
                "time_seconds": 0,
                "verification_valid": False,
            })
    
    # Print report
    report = metrics.get_report()
    print(report)
    
    # Save results
    if save_results:
        results_file = "evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump({
                "metrics": {
                    "citation_coverage": metrics.citation_coverage,
                    "eligibility_correctness": metrics.eligibility_correctness,
                    "abstention_accuracy": metrics.abstention_accuracy,
                },
                "individual_results": metrics.individual_results,
            }, f, indent=2)
        logger.info(f"Results saved to {results_file}")
    
    return metrics


if __name__ == "__main__":
    run_evaluation()
