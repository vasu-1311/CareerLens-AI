"""
CareerLens AI — Tool 4: Resume Rewriter
=========================================
Rewrites weak resume bullet points to be stronger,
more impactful, and ATS-friendly.
Uses rule-based templates (fast, no model download needed).
"""

import re
import random


# ── Power verbs by category ───────────────────────────────────
POWER_VERBS = {
    "built":       "Engineered",
    "made":        "Developed",
    "created":     "Designed and implemented",
    "worked on":   "Spearheaded",
    "helped":      "Contributed to",
    "did":         "Executed",
    "used":        "Leveraged",
    "implemented": "Architected and deployed",
    "developed":   "Built and deployed",
    "trained":     "Trained and optimised",
    "analysed":    "Performed in-depth analysis of",
    "tested":      "Validated and tested",
    "wrote":       "Authored",
}

# ── Impact phrases to append based on keywords ────────────────
IMPACT_PHRASES = {
    "machine learning":   "improving prediction accuracy and enabling data-driven decisions",
    "logistic regression":"achieving {acc}% classification accuracy on test data",
    "deep learning":      "reducing manual effort and improving model generalisation",
    "nlp":                "enabling automated text understanding at scale",
    "recommendation":     "increasing user engagement through personalised suggestions",
    "streamlit":          "enabling non-technical stakeholders to interact with ML outputs",
    "fastapi":            "serving real-time predictions via REST API endpoints",
    "data":               "extracting actionable insights from raw datasets",
    "eda":                "identifying key patterns and informing feature engineering decisions",
    "python":             "automating workflows and reducing manual processing time",
    "sql":                "enabling efficient querying across large relational datasets",
    "langchain":          "building production-ready LLM-powered applications",
    "rag":                "reducing hallucination and grounding responses in real documents",
    "classification":     "enabling automated categorisation with measurable accuracy",
    "clustering":         "revealing hidden patterns and segments in the data",
    "preprocessing":      "ensuring data quality and improving downstream model performance",
}


def upgrade_verb(sentence: str) -> str:
    """Replace weak verbs with stronger action verbs."""
    sentence_lower = sentence.lower()
    for weak, strong in POWER_VERBS.items():
        if sentence_lower.startswith(weak):
            return strong + sentence[len(weak):]
    return sentence


def add_impact(sentence: str, accuracy: int = None) -> str:
    """Add an impact phrase based on keywords in the sentence."""
    sentence_lower = sentence.lower()

    for keyword, phrase in IMPACT_PHRASES.items():
        if keyword in sentence_lower:
            # Add accuracy if available
            if "{acc}" in phrase:
                acc = accuracy if accuracy else random.randint(78, 92)
                phrase = phrase.format(acc=acc)

            # Only add if sentence doesn't already have impact language
            if not any(x in sentence_lower for x in ["%", "accuracy", "improving", "enabling", "reducing"]):
                if not sentence.endswith('.'):
                    sentence += '.'
                return sentence + f" This contributed to {phrase}."

    return sentence


def add_quantification(sentence: str) -> str:
    """Add quantification hints where missing."""
    sentence_lower = sentence.lower()

    # If no numbers present, suggest adding them
    has_numbers = bool(re.search(r'\d+', sentence))

    if not has_numbers:
        if "dataset" in sentence_lower or "data" in sentence_lower:
            return sentence.rstrip('.') + " on a dataset of 10,000+ records."
        if "model" in sentence_lower and "accuracy" not in sentence_lower:
            acc = random.randint(78, 92)
            return sentence.rstrip('.') + f", achieving {acc}% accuracy on the test set."

    return sentence


def rewrite_bullet(bullet: str, accuracy: int = None) -> str:
    """
    Full pipeline to rewrite a single resume bullet point.
    1. Upgrade verb
    2. Add quantification
    3. Add impact phrase
    """
    bullet = bullet.strip().lstrip('–').lstrip('-').strip()

    if len(bullet) < 10:
        return bullet

    # Step 1: Upgrade verb
    rewritten = upgrade_verb(bullet)

    # Step 2: Add quantification
    rewritten = add_quantification(rewritten)

    # Step 3: Add impact
    rewritten = add_impact(rewritten, accuracy)

    return rewritten


def rewrite_project_section(project_bullets: list, accuracy: int = None) -> list:
    """Rewrite a list of project bullet points."""
    return [rewrite_bullet(b, accuracy) for b in project_bullets if b.strip()]


def extract_bullets_from_text(text: str) -> list:
    """Extract bullet point lines from resume text."""
    bullets = []
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('–') or line.startswith('-') or line.startswith('•'):
            cleaned = line.lstrip('–').lstrip('-').lstrip('•').strip()
            if len(cleaned) > 20:
                bullets.append(cleaned)
    return bullets


class ResumeRewriter:
    """
    Rewrites weak resume sections using rule-based
    enhancement — no model download needed, instant results.
    """

    def rewrite_resume(self, full_text: str, projects: list, skills: list) -> dict:
        """
        Main rewrite method.
        Returns dict with rewritten sections.
        """
        # Extract bullet points from full text
        all_bullets = extract_bullets_from_text(full_text)

        # Rewrite bullets
        rewritten_bullets = []
        for bullet in all_bullets[:8]:  # top 8 bullet points
            original  = bullet
            rewritten = rewrite_bullet(bullet)
            if rewritten != original:
                rewritten_bullets.append({
                    "original":  original,
                    "rewritten": rewritten,
                })

        # Generate stronger skills section
        skills_section = self._rewrite_skills_section(skills)

        # Generate stronger project descriptions
        project_descriptions = self._generate_project_tips(projects, skills)

        return {
            "rewritten_bullets":     rewritten_bullets,
            "skills_section":        skills_section,
            "project_descriptions":  project_descriptions,
        }

    def _rewrite_skills_section(self, skills: list) -> str:
        """Generate a well-formatted skills section."""
        if not skills:
            return ""

        # Group skills by category
        categories = {
            "Languages & Libraries": [],
            "ML / AI":               [],
            "Tools & Platforms":     [],
            "Gen AI Stack":          [],
        }

        lang_skills  = {"python", "sql", "pandas", "numpy", "matplotlib", "seaborn", "opencv"}
        ml_skills    = {"machine learning", "deep learning", "nlp", "computer vision",
                        "scikit-learn", "tensorflow", "pytorch", "keras",
                        "linear regression", "logistic regression", "decision tree",
                        "random forest", "knn", "xgboost", "statistics",
                        "feature engineering", "eda", "data visualization"}
        tool_skills  = {"streamlit", "flask", "fastapi", "git", "github",
                        "jupyter", "docker", "aws"}
        genai_skills = {"langchain", "huggingface", "rag", "llm",
                        "prompt engineering", "embeddings", "faiss"}

        for skill in skills:
            s = skill.lower()
            if s in lang_skills:
                categories["Languages & Libraries"].append(skill.title())
            elif s in ml_skills:
                categories["ML / AI"].append(skill.title())
            elif s in genai_skills:
                categories["Gen AI Stack"].append(skill.title())
            elif s in tool_skills:
                categories["Tools & Platforms"].append(skill.title())
            else:
                categories["ML / AI"].append(skill.title())

        lines = []
        for cat, cat_skills in categories.items():
            if cat_skills:
                lines.append(f"{cat}: {', '.join(cat_skills)}")

        return "\n".join(lines)

    def _generate_project_tips(self, projects: list, skills: list) -> list:
        """Generate tips for making project descriptions stronger."""
        tips = []
        for project in projects[:4]:
            tip = {
                "project": project,
                "suggestions": [
                    f"Add the specific dataset size used (e.g., '10,000 records' or '50MB dataset')",
                    f"Include a quantified result (e.g., 'achieved 87% accuracy' or 'reduced processing time by 40%')",
                    f"Mention the specific tools used from your stack",
                    f"Add a GitHub link with a proper README explaining the project",
                ]
            }
            tips.append(tip)
        return tips


# ── Standalone test ───────────────────────────────────────────
if __name__ == "__main__":
    rewriter = ResumeRewriter()

    test_bullets = [
        "Built a machine learning model to predict house prices",
        "Used logistic regression for classification",
        "Created a recommendation system using Python",
        "Developed a streamlit app for data visualization",
    ]

    print("\nResume Rewriter Test:")
    for bullet in test_bullets:
        rewritten = rewrite_bullet(bullet)
        print(f"\n  Original : {bullet}")
        print(f"  Rewritten: {rewritten}")
