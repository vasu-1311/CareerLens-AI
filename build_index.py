"""
CareerLens AI — FAISS Index Builder
=====================================
Reads job_descriptions.csv
Embeds all JDs using all-MiniLM-L6-v2
Saves FAISS index to faiss_index/

Run: python build_index.py
"""

import pandas as pd
import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ── Paths ─────────────────────────────────────────────────────
CSV_PATH        = "data/job_descriptions.csv"
FAISS_PATH      = "faiss_index/jd_index"

def build_index():
    print("=" * 55)
    print("  CareerLens AI — FAISS Index Builder")
    print("=" * 55)

    # ── 1. Load CSV ───────────────────────────────────────────
    print("\n[1/4] Loading job descriptions...")
    df = pd.read_csv(CSV_PATH)

    # Clean up
    df = df.dropna(subset=["Full_JD_Text"])
    df = df[df["Full_JD_Text"].str.len() > 50]
    df = df.reset_index(drop=True)

    print(f"  Total JDs loaded : {len(df)}")
    print(f"  Domains          : {df['Domain'].value_counts().to_dict()}")

    # ── 2. Convert to LangChain Documents ────────────────────
    print("\n[2/4] Converting to documents...")
    documents = []
    for _, row in df.iterrows():
        # Combine all fields into one rich text for embedding
        content = f"""
Role: {row.get('Role', '')}
Company: {row.get('Company', '')}
Domain: {row.get('Domain', '')}
Experience: {row.get('Experience', '')}
Skills Required: {row.get('Skills', '')}
Job Description: {row.get('Full_JD_Text', '')}
        """.strip()

        doc = Document(
            page_content=content,
            metadata={
                "company":    str(row.get("Company",    "")),
                "role":       str(row.get("Role",       "")),
                "domain":     str(row.get("Domain",     "")),
                "experience": str(row.get("Experience", "")),
                "skills":     str(row.get("Skills",     "")),
                "type":       str(row.get("Type",       "")),
            }
        )
        documents.append(doc)

    print(f"  Documents created: {len(documents)}")

    # ── 3. Load Embedding Model ───────────────────────────────
    print("\n[3/4] Loading HuggingFace embedding model...")
    print("  Model: sentence-transformers/all-MiniLM-L6-v2")
    print("  (First run downloads ~90 MB — please wait...)")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── 4. Build and Save FAISS Index ────────────────────────
    print("\n[4/4] Building FAISS index and saving...")
    os.makedirs("faiss_index", exist_ok=True)

    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(FAISS_PATH)

    # Also save the raw dataframe for displaying results later
    df.to_pickle("faiss_index/jd_dataframe.pkl")

    print("\n" + "=" * 55)
    print(f"  DONE!")
    print(f"  FAISS index saved to : {FAISS_PATH}")
    print(f"  Total JDs indexed    : {len(documents)}")
    print("=" * 55)
    print("\n  Next step: python train_model.py")


if __name__ == "__main__":
    build_index()
