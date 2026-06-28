"""
CareerLens AI — Tool 5: Plan Generator
========================================
Generates a personalised 30-day action plan based on:
- Skill gaps identified by gap analyser
- Matched jobs from job searcher
- Candidate's current strength areas
No LLM needed — rule-based generation, instant results.
"""


# ── Learning resources per skill ─────────────────────────────
RESOURCES = {
    "python":           ("Python for Everybody — Coursera (free audit)",     "2-3 days"),
    "machine learning": ("Andrew Ng ML Course — Coursera (free audit)",      "1 week"),
    "deep learning":    ("fast.ai Practical Deep Learning — free",           "1 week"),
    "sql":              ("SQLZoo.net + HackerRank SQL track — free",         "3-4 days"),
    "tensorflow":       ("TensorFlow official tutorials — tensorflow.org",    "4-5 days"),
    "pytorch":          ("PyTorch 60-minute blitz — pytorch.org",            "3-4 days"),
    "nlp":              ("Hugging Face NLP Course — huggingface.co/learn",   "1 week"),
    "langchain":        ("LangChain docs + build a RAG chatbot",             "3-4 days"),
    "statistics":       ("Khan Academy Statistics — free",                    "1 week"),
    "docker":           ("Docker Getting Started guide — docs.docker.com",   "2 days"),
    "aws":              ("AWS Cloud Practitioner — free practice on ExamPro", "1 week"),
    "git":              ("learngitbranching.js.org — interactive, free",     "1-2 days"),
    "streamlit":        ("Streamlit docs + build an ML demo app",            "1-2 days"),
    "fastapi":          ("FastAPI official tutorial — fastapi.tiangolo.com", "2-3 days"),
    "pandas":           ("Kaggle Pandas course — free",                      "2 days"),
    "scikit-learn":     ("Scikit-learn official user guide + examples",      "3 days"),
    "huggingface":      ("Hugging Face course — huggingface.co/learn",       "4-5 days"),
    "rag":              ("Build a RAG pipeline with LangChain + FAISS",      "3-4 days"),
    "computer vision":  ("OpenCV tutorials + build an object detection demo", "1 week"),
    "feature engineering": ("Kaggle Feature Engineering course — free",      "3 days"),
}

# ── Project ideas per skill ───────────────────────────────────
PROJECT_IDEAS = {
    "machine learning":  "Build an end-to-end ML project: collect data, clean it, train a model, and deploy with Streamlit",
    "nlp":               "Build a sentiment analyser or text summariser on a public dataset from Kaggle",
    "deep learning":     "Train a CNN for image classification using CIFAR-10 dataset",
    "sql":               "Analyse a real dataset (try Kaggle) using SQL and build a simple dashboard",
    "langchain":         "Build a PDF Q&A chatbot using LangChain + FAISS — use your own resume as the document",
    "computer vision":   "Build a face mask detector or object counter using OpenCV + Python",
    "streamlit":         "Convert one of your existing ML models into an interactive Streamlit web app",
    "fastapi":           "Wrap your best ML model in a FastAPI endpoint and test it with Postman",
    "statistics":        "Perform a full EDA on any Kaggle dataset and document your statistical findings",
    "huggingface":       "Fine-tune a small HuggingFace model on a classification task and push to Hub",
}

# ── Company search terms for Bangalore ML fresher roles ──────
COMPANIES_TO_TARGET = [
    "Sarvam AI", "Sigmoid", "Hasura", "Postman", "Sprinklr",
    "Meesho", "Razorpay", "Zepto", "Swiggy AI team",
    "Flipkart AI", "PhonePe Data", "BrowserStack",
    "Delhivery Data Science", "Unacademy AI", "CRED Data",
]


class PlanGenerator:
    """
    Generates a personalised 30-day job-readiness action plan.
    """

    def generate(
        self,
        gap_result: dict,
        job_results: list,
        candidate_name: str = "Candidate",
    ) -> dict:
        """
        Generate full 30-day plan.

        Args:
            gap_result   : output from GapAnalyser.analyse()
            job_results  : output from JobSearcher.search()
            candidate_name: candidate's name for personalisation

        Returns:
            dict with week-by-week plan, target companies, interview tips
        """
        readiness       = gap_result.get("readiness_score", 50)
        critical_missing= gap_result.get("critical_missing", [])
        strengths       = gap_result.get("strengths", [])
        verdict         = gap_result.get("verdict", "Needs Work")
        tips            = gap_result.get("improvement_tips", [])

        # Top matched companies
        top_companies = list(set([
            r["company"] for r in job_results[:5]
            if r.get("company") and r["company"] != "Unknown"
        ]))

        # Build weekly plan based on readiness score
        if readiness >= 70:
            plan = self._plan_ready(critical_missing, strengths, top_companies)
        elif readiness >= 45:
            plan = self._plan_needs_work(critical_missing, strengths, top_companies)
        else:
            plan = self._plan_build_skills(critical_missing, strengths, top_companies)

        # Interview questions based on strengths
        interview_tips = self._generate_interview_tips(strengths, critical_missing)

        # Resources for top missing skills
        resources = []
        for skill in critical_missing[:4]:
            if skill in RESOURCES:
                name_r, duration = RESOURCES[skill]
                resources.append({
                    "skill":    skill.title(),
                    "resource": name_r,
                    "duration": duration,
                })

        # Project idea
        project_idea = ""
        for skill in critical_missing + strengths:
            if skill in PROJECT_IDEAS:
                project_idea = PROJECT_IDEAS[skill]
                break

        return {
            "candidate_name":   candidate_name,
            "readiness_score":  readiness,
            "verdict":          verdict,
            "weekly_plan":      plan,
            "target_companies": top_companies if top_companies else COMPANIES_TO_TARGET[:5],
            "resources":        resources,
            "project_idea":     project_idea,
            "interview_tips":   interview_tips,
            "improvement_tips": tips,
        }

    def _plan_ready(self, missing, strengths, companies) -> list:
        """Plan for candidates scoring 70+."""
        skill1 = missing[0].title() if missing else "interview skills"
        skill2 = missing[1].title() if len(missing) > 1 else "system design basics"
        cos    = ", ".join(companies[:3]) if companies else "top Bangalore ML startups"

        return [
            {
                "week":  "Week 1",
                "title": "Polish and Apply",
                "tasks": [
                    f"Update your resume with quantified results in every bullet point",
                    f"Create or update your GitHub profile — pin your best 3 projects",
                    f"Apply to {cos} on LinkedIn and their official careers pages",
                    f"Set up job alerts on LinkedIn and Naukri for ML Engineer fresher Bangalore",
                ]
            },
            {
                "week":  "Week 2",
                "title": "Fill Remaining Gaps",
                "tasks": [
                    f"Learn {skill1} — spend 1-2 hours daily this week",
                    f"Build a small project demonstrating {skill1} and push to GitHub",
                    f"Start learning {skill2} — even basics make a difference in interviews",
                    f"Write a LinkedIn post about a project you built — builds visibility",
                ]
            },
            {
                "week":  "Week 3",
                "title": "Interview Preparation",
                "tasks": [
                    f"Practice ML fundamentals: bias-variance tradeoff, overfitting, cross-validation",
                    f"Practice explaining your best project in exactly 3 minutes",
                    f"Do 5 LeetCode Easy problems in Python — shows basic coding ability",
                    f"Research each company you applied to — know their products",
                ]
            },
            {
                "week":  "Week 4",
                "title": "Follow Up and Expand",
                "tasks": [
                    f"Follow up on applications sent in Week 1 via LinkedIn message",
                    f"Apply to 10 more companies from Internshala and Wellfound",
                    f"Ask for referrals from college seniors or LinkedIn connections at target companies",
                    f"Keep building — one new small project this week",
                ]
            },
        ]

    def _plan_needs_work(self, missing, strengths, companies) -> list:
        """Plan for candidates scoring 45-70."""
        skill1   = missing[0].title() if missing else "SQL"
        skill2   = missing[1].title() if len(missing) > 1 else "Deep Learning"
        resource1= RESOURCES.get(missing[0].lower(), ("Official docs + tutorials", "1 week"))[0] if missing else ""
        cos      = ", ".join(companies[:3]) if companies else "Internshala ML/AI listings"

        return [
            {
                "week":  "Week 1",
                "title": f"Learn {skill1}",
                "tasks": [
                    f"Focus entirely on {skill1} this week — 2 hours every day",
                    f"Resource: {resource1}",
                    f"By end of week: build one small project using {skill1}",
                    f"Push the project to GitHub with a proper README",
                ]
            },
            {
                "week":  "Week 2",
                "title": f"Learn {skill2} + Update Resume",
                "tasks": [
                    f"Start learning {skill2} — 1 hour daily",
                    f"Rewrite your resume bullet points using the suggestions CareerLens gave you",
                    f"Make sure every project has: what you built, what tools, what result",
                    f"Update LinkedIn profile with new skills and projects",
                ]
            },
            {
                "week":  "Week 3",
                "title": "Build and Apply",
                "tasks": [
                    f"Build one end-to-end project combining your strongest skills",
                    f"Deploy it on Streamlit Cloud — get a live link",
                    f"Start applying to {cos}",
                    f"Apply to 10-15 internships and fresher roles this week",
                ]
            },
            {
                "week":  "Week 4",
                "title": "Interview Prep",
                "tasks": [
                    f"Practice core ML interview questions: explain your models, metrics, overfitting",
                    f"Practice coding: 3 Python problems daily on HackerRank",
                    f"Prepare a 2-minute introduction about yourself and your projects",
                    f"Follow up on all applications and expand to more platforms",
                ]
            },
        ]

    def _plan_build_skills(self, missing, strengths, companies) -> list:
        """Plan for candidates scoring below 45."""
        skill1 = missing[0].title() if missing else "Machine Learning"
        skill2 = missing[1].title() if len(missing) > 1 else "Python"

        return [
            {
                "week":  "Week 1",
                "title": "Core Skills Foundation",
                "tasks": [
                    f"Make sure Python is solid — complete Kaggle Python course (free, 5 hours)",
                    f"Start {skill1} fundamentals — Andrew Ng course or fast.ai",
                    f"Do EDA on one Kaggle dataset and document your findings",
                    f"Do not apply yet — build first, apply when stronger",
                ]
            },
            {
                "week":  "Week 2",
                "title": "Build First Real Project",
                "tasks": [
                    f"Build a complete ML project end-to-end: data → model → evaluation → Streamlit app",
                    f"Use a Kaggle dataset — Titanic, House Prices, or Heart Disease are good starters",
                    f"Push to GitHub with proper README explaining your approach",
                    f"Continue learning {skill2} — 1 hour daily",
                ]
            },
            {
                "week":  "Week 3",
                "title": "Expand Skills + Second Project",
                "tasks": [
                    f"Build a second project in a different area (NLP or Computer Vision)",
                    f"Deploy at least one project on Streamlit Cloud for a live link",
                    f"Update resume with both projects and quantified results",
                    f"Start applying to internships only — not full-time roles yet",
                ]
            },
            {
                "week":  "Week 4",
                "title": "Apply and Keep Building",
                "tasks": [
                    f"Apply to 20 internships on Internshala for ML/Data Science",
                    f"Apply to fresher roles on Naukri with 0-1 years filter",
                    f"Continue learning — one new skill per week going forward",
                    f"Join ML communities: Kaggle, Hugging Face Discord, LinkedIn ML groups",
                ]
            },
        ]

    def _generate_interview_tips(self, strengths: list, missing: list) -> list:
        """Generate interview preparation tips based on candidate profile."""
        tips = []

        QUESTION_TEMPLATES = {
            "machine learning":  "Be ready to explain: bias vs variance, overfitting solutions, cross-validation, and when to use which algorithm",
            "scikit-learn":      "Practice explaining your model evaluation — why you chose accuracy vs F1, what your confusion matrix shows",
            "deep learning":     "Know: what is backpropagation, vanishing gradient problem, dropout, batch normalisation",
            "nlp":               "Be ready to explain TF-IDF vs word embeddings, and how transformer models work at a high level",
            "langchain":         "Explain your RAG pipeline: chunking strategy, why you chose your chunk size, how FAISS similarity search works",
            "python":            "Practice writing clean functions, list comprehensions, and explain OOP concepts",
            "pandas":            "Know how to handle missing values, merge dataframes, and groupby operations",
            "statistics":        "Know: normal distribution, p-value, hypothesis testing, correlation vs causation",
        }

        for skill in strengths[:4]:
            if skill in QUESTION_TEMPLATES:
                tips.append({
                    "skill":    skill.title(),
                    "tip":      QUESTION_TEMPLATES[skill],
                })

        # General tips always included
        tips.append({
            "skill": "All Interviews",
            "tip":   "Always explain the PROBLEM first, then your approach, then results. Interviewers care about your thinking process more than the code.",
        })
        tips.append({
            "skill": "Project Explanation",
            "tip":   "Prepare a 2-minute walkthrough for each project: what problem, what data, what model/approach, what result, what you learned.",
        })

        return tips


# ── Standalone test ───────────────────────────────────────────
if __name__ == "__main__":
    generator = PlanGenerator()

    mock_gap = {
        "readiness_score":  62,
        "verdict":          "Needs Work",
        "critical_missing": ["sql", "deep learning", "docker"],
        "strengths":        ["python", "machine learning", "scikit-learn", "streamlit"],
        "improvement_tips": ["Learn SQL this week"],
    }
    mock_jobs = [
        {"company": "Sarvam AI",  "role": "ML Intern",    "match_score": 78},
        {"company": "Sigmoid",    "role": "Data Analyst", "match_score": 71},
    ]

    plan = generator.generate(mock_gap, mock_jobs, "Vasudev")

    print(f"\n30-Day Action Plan for {plan['candidate_name']}")
    print(f"Readiness: {plan['readiness_score']}/100 — {plan['verdict']}")
    for week in plan["weekly_plan"]:
        print(f"\n{week['week']}: {week['title']}")
        for task in week["tasks"]:
            print(f"  - {task}")
