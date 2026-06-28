"""
CareerLens AI — Tool 2: Job Searcher
======================================
Searches FAISS index for jobs that semantically
match the candidate's resume skills profile.
Returns ranked list of compatible jobs with match scores.
"""

import os
import joblib
import pandas as pd
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

FAISS_PATH = "faiss_index/jd_index"
DF_PATH    = "faiss_index/jd_dataframe.pkl"


class JobSearcher:
    """
    Loads pre-built FAISS index and searches for
    jobs matching the candidate's skill profile.
    """

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store = None
        self.jd_df        = None
        self._load_index()

    def _load_index(self):
        """Load FAISS index and JD dataframe from disk."""
        if os.path.exists(FAISS_PATH):
            self.vector_store = FAISS.load_local(
                FAISS_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        if os.path.exists(DF_PATH):
            self.jd_df = pd.read_pickle(DF_PATH)

    def search(self, skills: list, full_text: str = "", top_k: int = 10) -> list:
        """
        Search for matching jobs using candidate skills + resume text.
        Returns list of job dicts with match scores.
        """
        if not self.vector_store:
            return []

        # Build a rich query from skills + resume context
        skill_query = " ".join(skills)
        if full_text:
            # Use first 500 chars of resume as additional context
            query = f"ML AI internship fresher {skill_query} {full_text[:300]}"
        else:
            query = f"ML AI internship fresher {skill_query}"

        # Semantic search — get top_k * 2 then re-rank
        docs_scores = self.vector_store.similarity_search_with_score(
            query, k=min(top_k * 2, 50)
        )

        results = []
        seen    = set()

        for doc, score in docs_scores:
            meta    = doc.metadata
            company = meta.get("company", "")
            role    = meta.get("role", "")

            # Deduplicate by company + role
            key = (company.lower(), role.lower())
            if key in seen:
                continue
            seen.add(key)

            # Convert FAISS L2 distance to similarity score (0-100)
            # Lower L2 distance = better match
            match_pct = max(0, round((1 - score / 2) * 100, 1))
            match_pct = min(match_pct, 99)  # cap at 99%

            # Skill overlap score
            jd_skills_raw = meta.get("skills", "").lower()
            overlap = sum(1 for s in skills if s.lower() in jd_skills_raw)
            skill_overlap_pct = round((overlap / len(skills) * 100) if skills else 0, 1)

            # Combined score (70% semantic + 30% skill overlap)
            combined = round(0.70 * match_pct + 0.30 * skill_overlap_pct, 1)

            results.append({
                "company":          company,
                "role":             role,
                "domain":           meta.get("domain", "ML / AI"),
                "experience":       meta.get("experience", "Fresher"),
                "skills_required":  meta.get("skills", ""),
                "type":             meta.get("type", "Internship"),
                "match_score":      combined,
                "semantic_score":   match_pct,
                "skill_overlap":    skill_overlap_pct,
                "jd_preview":       doc.page_content[:300],
            })

        # Sort by combined score descending
        results = sorted(results, key=lambda x: x["match_score"], reverse=True)
        return results[:top_k]


# ── Standalone test ───────────────────────────────────────────
if __name__ == "__main__":
    searcher = JobSearcher()
    test_skills = ["python", "machine learning", "scikit-learn", "pandas", "streamlit"]
    results = searcher.search(test_skills, top_k=5)

    print("\nTop Matching Jobs:")
    for i, r in enumerate(results, 1):
        print(f"\n  {i}. {r['role']} at {r['company']}")
        print(f"     Match Score : {r['match_score']}%")
        print(f"     Skills      : {r['skills_required'][:80]}")
