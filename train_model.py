"""
CareerLens AI — ML Gap Classifier Training
===========================================
Generates synthetic candidate profiles based on
real ML/AI skills extracted from job_descriptions.csv
Trains Logistic Regression + Decision Tree + KNN classifiers
Saves best model to models/gap_classifier.pkl

Run: python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

np.random.seed(42)
os.makedirs("models", exist_ok=True)

print("=" * 55)
print("  CareerLens AI — ML Gap Classifier Training")
print("=" * 55)

# ── 1. Define ML/AI Skill Universe ───────────────────────────
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

HIGH_IMPACT = [
    "python", "machine learning", "scikit-learn",
    "sql", "deep learning", "tensorflow", "pytorch",
    "nlp", "langchain", "huggingface", "rag",
    "statistics", "feature engineering",
]

MED_IMPACT = [
    "pandas", "numpy", "eda", "data visualization",
    "git", "streamlit", "flask", "docker",
    "linear regression", "logistic regression",
    "random forest", "xgboost",
]

print("\n[1/5] Generating synthetic candidate profiles...")

profiles = []
for i in range(400):
    candidate_type = np.random.choice(
        ["strong", "moderate", "weak"], p=[0.35, 0.40, 0.25]
    )

    if candidate_type == "strong":
        hi = np.random.choice(HIGH_IMPACT, size=np.random.randint(6, min(11, len(HIGH_IMPACT))), replace=False).tolist()
        md = np.random.choice(MED_IMPACT,  size=np.random.randint(4, min(7,  len(MED_IMPACT))),  replace=False).tolist()
        skills = list(set(hi + md))
        shortlisted = 1 if np.random.random() < 0.82 else 0

    elif candidate_type == "moderate":
        hi = np.random.choice(HIGH_IMPACT, size=np.random.randint(3, min(6, len(HIGH_IMPACT))), replace=False).tolist()
        md = np.random.choice(MED_IMPACT,  size=np.random.randint(2, min(5, len(MED_IMPACT))),  replace=False).tolist()
        skills = list(set(hi + md))
        shortlisted = 1 if np.random.random() < 0.45 else 0

    else:
        hi = np.random.choice(HIGH_IMPACT, size=np.random.randint(1, 3), replace=False).tolist()
        md = np.random.choice(MED_IMPACT,  size=np.random.randint(0, 3), replace=False).tolist()
        skills = list(set(hi + md))
        shortlisted = 1 if np.random.random() < 0.15 else 0

    profiles.append({"skills": skills, "shortlisted": shortlisted})

df_profiles = pd.DataFrame(profiles)
print(f"  Profiles generated : {len(df_profiles)}")
print(f"  Shortlisted (1)    : {df_profiles['shortlisted'].sum()}")
print(f"  Not shortlisted (0): {(df_profiles['shortlisted']==0).sum()}")

print("\n[2/5] Engineering features...")
mlb = MultiLabelBinarizer(classes=ALL_SKILLS)
X   = mlb.fit_transform(df_profiles["skills"])
y   = df_profiles["shortlisted"].values
joblib.dump(mlb, "models/skill_binarizer.pkl")
print(f"  Feature vector size: {X.shape[1]} skills")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n[3/5] Training models...")
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=500),
    "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=7),
}

results = {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    cv     = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1")

    results[name] = {
        "accuracy":   round(accuracy_score(y_test, y_pred), 4),
        "precision":  round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":     round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":   round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":    round(roc_auc_score(y_test, y_prob), 4),
        "cv_f1_mean": round(cv.mean(), 4),
    }
    print(f"\n  [{name}]")
    print(f"    Accuracy : {results[name]['accuracy']:.2%}")
    print(f"    F1 Score : {results[name]['f1_score']:.2%}")
    print(f"    ROC-AUC  : {results[name]['roc_auc']:.2%}")

print("\n[4/5] Selecting and saving best model...")
best_name = max(results, key=lambda k: results[k]["roc_auc"])
best_clf  = models[best_name]
print(f"  Best model : {best_name} (ROC-AUC: {results[best_name]['roc_auc']:.2%})")
joblib.dump(best_clf, "models/gap_classifier.pkl")

print("\n[5/5] Saving skill metadata...")
lr_model = models["Logistic Regression"]
skill_weights = dict(zip(mlb.classes_, lr_model.coef_[0]))
skill_weights_sorted = dict(sorted(skill_weights.items(), key=lambda x: x[1], reverse=True))

export = {
    "best_model":    best_name,
    "all_skills":    ALL_SKILLS,
    "high_impact":   HIGH_IMPACT,
    "model_results": results,
    "skill_weights": skill_weights_sorted,
}
with open("models/model_metadata.json", "w") as f:
    json.dump(export, f, indent=2)

print("\n" + "=" * 55)
print("  DONE!")
print("  Saved: models/gap_classifier.pkl")
print("  Saved: models/skill_binarizer.pkl")
print("  Saved: models/model_metadata.json")
print("=" * 55)
print("\n  Next step: build the tools/ files")
