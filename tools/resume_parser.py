"""
CareerLens AI — Tool 1: Resume Parser
=======================================
Extracts structured information from a PDF resume:
- Candidate name, email, phone
- Skills detected (matched against ML/AI skill universe)
- Education details
- Projects listed
- Experience level
"""

import re
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ── Skill Universe (same as train_model.py) ───────────────────
ALL_SKILLS = [
    "python", "pandas", "numpy", "matplotlib", "seaborn",
    "scikit-learn", "tensorflow", "pytorch", "keras",
    "machine learning", "deep learning", "nlp",
    "computer vision", "sql", "statistics",
    "feature engineering", "eda", "data visualization",
    "langchain", "huggingface", "rag", "llm",
    "prompt engineering", "embeddings", "faiss",
    "streamlit", "flask", "fastapi",
    "git", "github", "jupyter",
    "docker", "aws", "opencv",
    "linear regression", "logistic regression",
    "decision tree", "random forest", "knn", "xgboost",
]

# Skill aliases — maps variations to standard names
SKILL_ALIASES = {
    "sklearn":              "scikit-learn",
    "sci-kit learn":        "scikit-learn",
    "tf":                   "tensorflow",
    "pytorch":              "pytorch",
    "torch":                "pytorch",
    "natural language":     "nlp",
    "natural language processing": "nlp",
    "large language model": "llm",
    "llms":                 "llm",
    "retrieval augmented":  "rag",
    "generative ai":        "llm",
    "gen ai":               "llm",
    "hf":                   "huggingface",
    "hugging face":         "huggingface",
    "k-nearest":            "knn",
    "k nearest":            "knn",
    "xgb":                  "xgboost",
    "random forests":       "random forest",
    "decision trees":       "decision tree",
    "exploratory data":     "eda",
    "data science":         "machine learning",
    "ml":                   "machine learning",
    "dl":                   "deep learning",
    "cv":                   "computer vision",
    "github":               "git",
}


def extract_skills(text: str) -> list:
    """Extract skills from resume text using pattern matching."""
    text_lower = text.lower()

    # Apply aliases first
    for alias, standard in SKILL_ALIASES.items():
        text_lower = text_lower.replace(alias, standard)

    found = []
    for skill in ALL_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill)

    return list(set(found))


def extract_name(text: str) -> str:
    """Extract candidate name from first few lines of resume."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    # Name is usually in the first 3 lines, all caps or title case
    for line in lines[:5]:
        # Skip lines with email, phone, or URLs
        if any(x in line.lower() for x in ['@', 'http', 'linkedin', 'github', '+91', 'phone']):
            continue
        # Skip very long lines (not a name)
        if len(line) > 50:
            continue
        # If it looks like a name (2-4 words, mostly letters)
        words = line.split()
        if 2 <= len(words) <= 4 and all(w.replace('.','').isalpha() for w in words):
            return line.title()
    return "Candidate"


def extract_email(text: str) -> str:
    """Extract email address from resume text."""
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    """Extract phone number from resume text."""
    match = re.search(r'(\+91[\s-]?)?[6-9]\d{9}', text)
    return match.group(0) if match else ""


def extract_education(text: str) -> list:
    """Extract education details."""
    edu_keywords = ["b.tech", "btech", "b.e", "be ", "bca", "mca", "m.tech",
                    "mtech", "mba", "bsc", "msc", "bachelor", "master",
                    "university", "college", "institute", "iit", "nit"]
    lines = text.split('\n')
    edu_lines = []
    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in edu_keywords):
            cleaned = line.strip()
            if cleaned and len(cleaned) > 5:
                edu_lines.append(cleaned)
    return edu_lines[:4]  # max 4 education entries


def extract_projects(text: str) -> list:
    """Extract project names from resume."""
    projects = []
    lines = text.split('\n')

    in_projects = False
    for line in lines:
        line_stripped = line.strip()
        line_lower    = line_stripped.lower()

        # Detect projects section
        if any(kw in line_lower for kw in ['project', 'projects', 'work done', 'portfolio']):
            in_projects = True
            continue

        # Stop at next major section
        if in_projects and any(kw in line_lower for kw in
            ['experience', 'education', 'certification', 'skill', 'achievement', 'award']):
            in_projects = False

        # Collect project titles (usually short lines, title case)
        if in_projects and line_stripped:
            if 5 < len(line_stripped) < 80:
                if not line_stripped.startswith('–') and not line_stripped.startswith('-'):
                    projects.append(line_stripped)

    return projects[:6]  # max 6 projects


def get_experience_level(text: str) -> str:
    """Determine experience level from resume text."""
    text_lower = text.lower()

    # Check for experience indicators
    if any(x in text_lower for x in ['fresher', 'fresh graduate', '0 years', 'no experience']):
        return "Fresher"

    # Look for year mentions
    year_match = re.findall(r'(\d+)\+?\s*years?\s*(of\s*)?(experience|exp)', text_lower)
    if year_match:
        years = int(year_match[0][0])
        if years == 0:
            return "Fresher"
        elif years <= 2:
            return f"{years} year(s) experience"
        else:
            return f"{years}+ years experience"

    # Check for internship
    if 'intern' in text_lower:
        return "Internship / Fresher"

    return "Fresher"


class ResumeParser:
    """
    Main resume parser class.
    Loads PDF, extracts structured info, builds internal FAISS index
    for Q&A retrieval.
    """

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store  = None
        self.full_text     = ""
        self.parsed_data   = {}

    def parse(self, pdf_path: str) -> dict:
        """
        Main method. Parses PDF and returns structured profile dict.
        """
        # Load PDF
        loader    = PyPDFLoader(pdf_path)
        documents = loader.load()
        self.full_text = "\n".join([d.page_content for d in documents])

        # Build internal FAISS for Q&A
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)

        # Extract all info
        skills    = extract_skills(self.full_text)
        name      = extract_name(self.full_text)
        email     = extract_email(self.full_text)
        phone     = extract_phone(self.full_text)
        education = extract_education(self.full_text)
        projects  = extract_projects(self.full_text)
        exp_level = get_experience_level(self.full_text)

        self.parsed_data = {
            "name":            name,
            "email":           email,
            "phone":           phone,
            "skills":          skills,
            "education":       education,
            "projects":        projects,
            "experience_level": exp_level,
            "full_text":       self.full_text,
            "skill_count":     len(skills),
        }

        return self.parsed_data

    def retrieve_context(self, query: str, k: int = 3) -> str:
        """Semantic retrieval from resume for Q&A."""
        if not self.vector_store:
            return "No resume loaded."
        docs = self.vector_store.similarity_search(query, k=k)
        return "\n---\n".join([d.page_content for d in docs])

    def extractive_summary(self) -> str:
        """Fast extractive summary of resume — no model needed."""
        if not self.full_text:
            return "No resume loaded."

        sentences = [s.strip() for s in re.split(r'[\.\n]', self.full_text)
                     if len(s.strip()) > 40]

        from collections import Counter
        all_words = re.findall(r'\b[a-zA-Z]{4,}\b', self.full_text.lower())
        word_freq = Counter(all_words)
        stop = {"with","that","this","from","have","been","will","they",
                "their","which","also","into","more","about","your","some",
                "using","used","work","worked","based","able","good"}

        def score(sent):
            words = re.findall(r'\b[a-zA-Z]{4,}\b', sent.lower())
            return sum(word_freq[w] for w in words if w not in stop)

        scored  = sorted(sentences, key=score, reverse=True)
        top     = set(scored[:6])
        ordered = [s for s in sentences if s in top]
        return " ".join(ordered[:6])


# ── Standalone test ───────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        parser = ResumeParser()
        result = parser.parse(sys.argv[1])
        print("\nParsed Resume:")
        print(f"  Name      : {result['name']}")
        print(f"  Email     : {result['email']}")
        print(f"  Skills    : {result['skills']}")
        print(f"  Projects  : {result['projects']}")
        print(f"  Education : {result['education']}")
        print(f"  Exp Level : {result['experience_level']}")
    else:
        print("Usage: python tools/resume_parser.py path/to/resume.pdf")
