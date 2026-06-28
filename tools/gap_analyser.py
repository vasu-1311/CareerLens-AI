"""
CareerLens AI — Tool 3: Gap Analyser
======================================
Uses trained Logistic Regression model to:
- Predict shortlisting probability
- Calculate readiness score
- Identify critical missing skills
- Rank skills by impact on shortlisting
"""

import joblib
import json
import numpy as np

MODEL_PATH    = "models/gap_classifier.pkl"
BINARIZER_PATH= "models/skill_binarizer.pkl"
METADATA_PATH = "models/model_metadata.json"


class GapAnalyser:
    """
    ML-powered skill gap analyser.
    Uses trained classifier to predict shortlisting probability
    and identify which skills matter most.
    """

    def __init__(self):
        self.model      = joblib.load(MODEL_PATH)
        self.binarizer  = joblib.load(BINARIZER_PATH)

        with open(METADATA_PATH) as f:
            self.metadata = json.load(f)

        self.all_skills    = self.metadata["all_skills"]
        self.high_impact   = self.metadata["high_impact"]
        self.skill_weights = self.metadata["skill_weights"]

    def analyse(self, candidate_skills: list) -> dict:
        """
        Analyse skill gaps for a candidate.

        Returns:
            readiness_score   : 0-100 overall score
            shortlist_prob    : probability of shortlisting (%)
            verdict           : Ready / Needs Work / Not Ready
            critical_missing  : high impact skills not in resume
            good_to_have      : medium impact skills not in resume
            strengths         : strong skills candidate already has
            skill_importance  : ranked list of missing skills by importance
            improvement_tips  : specific actionable tips
        """
        # Normalise candidate skills to lowercase
        candidate_lower = [s.lower().strip() for s in candidate_skills]

        # Encode skills as binary feature vector
        X = self.binarizer.transform([candidate_lower])

        # Predict shortlisting probability
        prob        = self.model.predict_proba(X)[0]
        shortlist_p = round(prob[1] * 100, 1)

        # Readiness score — weighted combination
        # 60% from model probability + 40% from skill coverage
        skill_coverage = len([s for s in candidate_lower if s in self.all_skills])
        coverage_pct   = (skill_coverage / len(self.all_skills)) * 100

        readiness = round(0.60 * shortlist_p + 0.40 * coverage_pct, 1)
        readiness = min(readiness, 98)  # cap at 98

        # Critical missing — high impact skills not in resume
        critical_missing = [
            s for s in self.high_impact
            if s not in candidate_lower
        ]

        # Sort by skill weight (most important first)
        critical_missing = sorted(
            critical_missing,
            key=lambda s: self.skill_weights.get(s, 0),
            reverse=True
        )

        # Good to have — all skills not in resume, excluding critical
        all_missing = [s for s in self.all_skills if s not in candidate_lower]
        good_to_have = [s for s in all_missing if s not in critical_missing]
        good_to_have = sorted(
            good_to_have,
            key=lambda s: self.skill_weights.get(s, 0),
            reverse=True
        )[:8]

        # Strengths — skills candidate has, sorted by importance
        strengths = [s for s in candidate_lower if s in self.all_skills]
        strengths = sorted(
            strengths,
            key=lambda s: self.skill_weights.get(s, 0),
            reverse=True
        )

        # Verdict
        if readiness >= 70:
            verdict = "Ready to Apply"
        elif readiness >= 45:
            verdict = "Needs Work"
        else:
            verdict = "Build More Skills"

        # Skill importance list (top missing skills with weights)
        skill_importance = [
            {
                "skill":    s,
                "weight":   round(self.skill_weights.get(s, 0), 3),
                "priority": "Critical" if s in self.high_impact else "Good to Have",
            }
            for s in (critical_missing + good_to_have)[:10]
        ]

        # Improvement tips
        tips = _generate_tips(critical_missing, strengths, readiness)

        return {
            "readiness_score":   readiness,
            "shortlist_prob":    shortlist_p,
            "verdict":           verdict,
            "critical_missing":  critical_missing[:6],
            "good_to_have":      good_to_have[:5],
            "strengths":         strengths,
            "skill_importance":  skill_importance,
            "improvement_tips":  tips,
            "skill_count":       len(strengths),
            "total_skills":      len(self.all_skills),
        }


def _generate_tips(missing: list, strengths: list, score: float) -> list:
    """Generate specific actionable improvement tips."""
    tips = []

    SKILL_RESOURCES = {
        "sql":              "Practice SQL on SQLZoo.net or HackerRank SQL track — focus on JOINs and GROUP BY",
        "machine learning": "Complete Andrew Ng's ML course on Coursera (audit free) and build 2 projects",
        "deep learning":    "Start with fast.ai Practical Deep Learning — free and project focused",
        "tensorflow":       "Build a simple image classifier using TensorFlow — add it as a project on GitHub",
        "pytorch":          "Complete PyTorch official 60-minute blitz tutorial, then build a project",
        "nlp":              "Work through Hugging Face NLP course — completely free, very practical",
        "langchain":        "Build a simple RAG chatbot using LangChain docs — add it to your GitHub",
        "docker":           "Complete Docker's official Getting Started guide — takes 2 hours",
        "aws":              "Get AWS Cloud Practitioner certification — free practice exams available online",
        "statistics":       "Study statistics for ML on Khan Academy — focus on probability and distributions",
        "git":              "Learn Git branching on learngitbranching.js.org — interactive and free",
        "fastapi":          "Build a REST API for one of your existing ML models using FastAPI",
    }

    # Tip 1 — most critical missing skill
    if missing:
        top_missing = missing[0]
        resource = SKILL_RESOURCES.get(
            top_missing,
            f"Learn {top_missing.title()} and build a small project demonstrating it"
        )
        tips.append(f"Priority skill to add: {top_missing.title()} — {resource}")

    # Tip 2 — second most critical
    if len(missing) > 1:
        second = missing[1]
        resource = SKILL_RESOURCES.get(
            second,
            f"Add {second.title()} to your skill set with a hands-on project"
        )
        tips.append(f"Second priority: {second.title()} — {resource}")

    # Tip 3 — project advice
    if strengths:
        top_strength = strengths[0]
        tips.append(
            f"You have {top_strength.title()} — make sure your GitHub has at least "
            f"2 projects that clearly demonstrate this skill with good README files"
        )

    # Tip 4 — score based advice
    if score < 45:
        tips.append(
            "Your profile needs significant skill additions. Focus on Python + "
            "Machine Learning + one specialisation (NLP or Computer Vision) before applying"
        )
    elif score < 70:
        tips.append(
            "You are close. Add 2-3 more skills from the critical missing list "
            "and update your resume projects to show real impact and results"
        )
    else:
        tips.append(
            "Strong profile. Focus on applying to companies now and preparing "
            "for technical interviews — practice ML fundamentals and system design basics"
        )

    return tips


# ── Standalone test ───────────────────────────────────────────
if __name__ == "__main__":
    analyser = GapAnalyser()
    test_skills = ["python", "pandas", "scikit-learn", "machine learning", "streamlit", "git"]
    result = analyser.analyse(test_skills)

    print("\nGap Analysis Result:")
    print(f"  Readiness Score  : {result['readiness_score']}/100")
    print(f"  Shortlist Prob   : {result['shortlist_prob']}%")
    print(f"  Verdict          : {result['verdict']}")
    print(f"  Critical Missing : {result['critical_missing']}")
    print(f"  Strengths        : {result['strengths']}")
    print("\nImprovement Tips:")
    for i, tip in enumerate(result['improvement_tips'], 1):
        print(f"  {i}. {tip}")
