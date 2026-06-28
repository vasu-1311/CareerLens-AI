"""
CareerLens AI — Agent
======================
LangChain-powered agent that orchestrates all 5 tools:
1. Resume Parser    → extract skills and info
2. Job Searcher     → find matching jobs from FAISS
3. Gap Analyser     → ML-based skill gap scoring
4. Resume Rewriter  → improve weak bullet points
5. Plan Generator   → personalised 30-day action plan

The agent decides which tools to call and in what order
based on what it finds in each step.

Run: called from app.py
"""

import os
import sys
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.resume_parser  import ResumeParser
from tools.job_searcher   import JobSearcher
from tools.gap_analyser   import GapAnalyser
from tools.resume_rewriter import ResumeRewriter
from tools.plan_generator  import PlanGenerator


class CareerLensAgent:
    """
    Main agent class. Orchestrates all tools in the right order.
    Adapts the pipeline based on candidate's readiness score.
    """

    def __init__(self):
        print("Initialising CareerLens Agent...")
        self.parser   = ResumeParser()
        self.searcher = JobSearcher()
        self.analyser = GapAnalyser()
        self.rewriter = ResumeRewriter()
        self.planner  = PlanGenerator()
        print("Agent ready.")

    def run(self, pdf_path: str) -> dict:
        """
        Main pipeline. Accepts a PDF resume path.
        Returns complete analysis dict.

        Agent reasoning:
        1. Always parse resume first
        2. Always run gap analysis
        3. Search for jobs
        4. If readiness < 70 → rewrite resume sections
        5. Always generate action plan
        6. Compile full report
        """

        results = {}

        # ── Step 1: Parse Resume ─────────────────────────────
        print("\n[Agent] Step 1: Parsing resume...")
        parsed = self.parser.parse(pdf_path)
        results["parsed"] = parsed

        name   = parsed.get("name", "Candidate")
        skills = parsed.get("skills", [])
        print(f"  Name detected    : {name}")
        print(f"  Skills extracted : {len(skills)} — {skills[:5]}...")

        # ── Step 2: Gap Analysis (ML) ────────────────────────
        print("\n[Agent] Step 2: Running ML gap analysis...")
        gap = self.analyser.analyse(skills)
        results["gap"] = gap

        readiness = gap.get("readiness_score", 0)
        verdict   = gap.get("verdict", "")
        print(f"  Readiness Score : {readiness}/100")
        print(f"  Verdict         : {verdict}")
        print(f"  Critical Missing: {gap.get('critical_missing', [])}")

        # ── Step 3: Job Search ───────────────────────────────
        print("\n[Agent] Step 3: Searching matching jobs...")
        full_text = parsed.get("full_text", "")
        jobs = self.searcher.search(skills, full_text=full_text, top_k=10)
        results["jobs"] = jobs

        if jobs:
            print(f"  Found {len(jobs)} matching jobs")
            print(f"  Top match: {jobs[0]['role']} at {jobs[0]['company']} ({jobs[0]['match_score']}%)")
        else:
            print("  No jobs found — check FAISS index")

        # ── Step 4: Resume Rewrite (conditional) ─────────────
        # Agent decision: only rewrite if readiness < 75
        if readiness < 75:
            print(f"\n[Agent] Step 4: Readiness {readiness} < 75 → Rewriting resume sections...")
            rewrite = self.rewriter.rewrite_resume(
                full_text=full_text,
                projects=parsed.get("projects", []),
                skills=skills,
            )
            results["rewrite"] = rewrite
            print(f"  Rewritten bullets: {len(rewrite.get('rewritten_bullets', []))}")
        else:
            print(f"\n[Agent] Step 4: Readiness {readiness} >= 75 → Skipping rewrite (profile strong enough)")
            results["rewrite"] = {
                "rewritten_bullets":    [],
                "skills_section":       self.rewriter._rewrite_skills_section(skills),
                "project_descriptions": [],
            }

        # ── Step 5: Generate Action Plan ────────────────────
        print("\n[Agent] Step 5: Generating 30-day action plan...")
        plan = self.planner.generate(
            gap_result=gap,
            job_results=jobs,
            candidate_name=name,
        )
        results["plan"] = plan
        print(f"  Plan generated for: {name}")
        print(f"  Target companies  : {plan.get('target_companies', [])[:3]}")

        # ── Step 6: Resume Q&A setup ─────────────────────────
        # Parser already built internal FAISS index for Q&A
        results["resume_qa_ready"] = True

        # ── Compile Summary ──────────────────────────────────
        results["summary"] = {
            "name":             name,
            "skills_found":     len(skills),
            "readiness_score":  readiness,
            "verdict":          verdict,
            "jobs_found":       len(jobs),
            "top_job":          jobs[0] if jobs else None,
            "critical_missing": gap.get("critical_missing", []),
            "strengths":        gap.get("strengths", []),
        }

        print("\n[Agent] Analysis complete.")
        return results

    def answer_question(self, question: str) -> str:
        """Answer a question about the resume using RAG retrieval."""
        return self.parser.retrieve_context(question, k=3)

    def get_summary(self) -> str:
        """Get extractive summary of the loaded resume."""
        return self.parser.extractive_summary()


# ── Standalone test ───────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        agent   = CareerLensAgent()
        results = agent.run(sys.argv[1])
        summary = results["summary"]

        print("\n" + "=" * 50)
        print("  CAREERLENS ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"  Candidate       : {summary['name']}")
        print(f"  Skills Found    : {summary['skills_found']}")
        print(f"  Readiness Score : {summary['readiness_score']}/100")
        print(f"  Verdict         : {summary['verdict']}")
        print(f"  Jobs Matched    : {summary['jobs_found']}")
        print(f"  Critical Missing: {summary['critical_missing']}")
    else:
        print("Usage: python agent.py path/to/resume.pdf")
