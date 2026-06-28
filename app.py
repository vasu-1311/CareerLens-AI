"""
CareerLens AI — Main Streamlit App
====================================
4 pages:
  1. Analysis Dashboard  — readiness score, skills, matched jobs
  2. Resume Improvement  — rewritten bullets, better skills section
  3. 30 Day Action Plan  — weekly tasks, resources, target companies
  4. Resume Q&A          — ask anything about the resume

Run: streamlit run app.py
"""

import sys
sys.modules["torch.classes"] = None

import os
import tempfile
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="CareerLens AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0a0f1e; }

  section[data-testid="stSidebar"] {
    background: #0f1729;
    border-right: 1px solid #1a2540;
  }
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] label { color: #8899bb !important; }

  header[data-testid="stHeader"] { display: none; }

  .pg-title {
    font-size: 1.7rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.1rem;
  }
  .pg-sub {
    font-size: 0.9rem;
    color: #4a5568;
    margin-bottom: 1.6rem;
  }

  .card {
    background: #0f1729;
    border: 1px solid #1a2540;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
  }
  .card-accent {
    background: #0f1729;
    border: 1px solid #1a2540;
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
  }
  .card-green {
    background: #0f1729;
    border: 1px solid #1a2540;
    border-left: 4px solid #22c55e;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
  }
  .card-yellow {
    background: #0f1729;
    border: 1px solid #1a2540;
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
  }
  .card-red {
    background: #0f1729;
    border: 1px solid #1a2540;
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
  }

  .pill-green {
    display:inline-block; background:#14532d; color:#86efac;
    border-radius:20px; padding:3px 12px; font-size:0.8rem; margin:3px;
  }
  .pill-red {
    display:inline-block; background:#450a0a; color:#fca5a5;
    border-radius:20px; padding:3px 12px; font-size:0.8rem; margin:3px;
  }
  .pill-blue {
    display:inline-block; background:#1e3a5f; color:#93c5fd;
    border-radius:20px; padding:3px 12px; font-size:0.8rem; margin:3px;
  }
  .pill-yellow {
    display:inline-block; background:#451a03; color:#fcd34d;
    border-radius:20px; padding:3px 12px; font-size:0.8rem; margin:3px;
  }

  div[data-testid="metric-container"] {
    background: #0f1729;
    border: 1px solid #1a2540;
    border-radius: 10px;
    padding: 0.8rem 1rem;
  }
  div[data-testid="metric-container"] label { color: #4a5568 !important; }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e2e8f0 !important; }

  div.stButton > button {
    background: #1a2540;
    color: #e2e8f0;
    border: 1px solid #2d3f6b;
    border-radius: 8px;
    padding: 0.5rem 1.6rem;
    font-size: 0.9rem;
  }
  div.stButton > button:hover {
    background: #2d3f6b;
    border-color: #3b82f6;
  }

  .week-header {
    font-size: 1rem;
    font-weight: 700;
    color: #3b82f6;
    margin-bottom: 0.4rem;
  }
  .task-item {
    color: #94a3b8;
    font-size: 0.88rem;
    padding: 4px 0;
    border-bottom: 1px solid #1a2540;
  }

  .score-big {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1;
    text-align: center;
  }

  .original-text {
    color: #ef4444;
    font-size: 0.85rem;
    background: #1a0a0a;
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 4px;
  }
  .rewritten-text {
    color: #86efac;
    font-size: 0.85rem;
    background: #0a1a0a;
    border-radius: 6px;
    padding: 8px 12px;
  }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent    = None
    st.session_state.results  = None
    st.session_state.analyzed = False


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## CareerLens AI")
    st.markdown("<p style='color:#4a5568;font-size:0.82rem'>AI Career Coach for Freshers</p>", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Analysis Dashboard", "Resume Improvement", "30 Day Action Plan", "Resume Q&A"],
        label_visibility="collapsed",
        disabled=not st.session_state.analyzed,
    )

    st.markdown("---")

    if st.session_state.analyzed:
        summary = st.session_state.results.get("summary", {})
        score   = summary.get("readiness_score", 0)
        color   = "#22c55e" if score >= 70 else "#f59e0b" if score >= 45 else "#ef4444"
        st.markdown(
            f"<p style='color:{color};font-size:1.1rem;font-weight:700;text-align:center'>"
            f"{score}/100</p>"
            f"<p style='color:#4a5568;font-size:0.78rem;text-align:center'>Readiness Score</p>",
            unsafe_allow_html=True
        )
        st.markdown(f"<p style='color:#22c55e;font-size:0.82rem'>Candidate: <b>{summary.get('name','')}</b></p>", unsafe_allow_html=True)

        if st.button("Analyse New Resume"):
            st.session_state.agent    = None
            st.session_state.results  = None
            st.session_state.analyzed = False
            st.rerun()


# ── Upload Screen (shown before analysis) ─────────────────────
if not st.session_state.analyzed:
    st.markdown("<div class='pg-title'>CareerLens AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='pg-sub'>Upload your resume and get instant AI-powered career analysis — built for ML and AI freshers.</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**What you will get:**")
    st.markdown("""
    <ul style='color:#64748b;font-size:0.9rem;line-height:2'>
      <li>Readiness score out of 100 based on your skills</li>
      <li>Top 10 matching ML/AI fresher jobs from real listings</li>
      <li>Exact skills missing from your resume with priority ranking</li>
      <li>Rewritten resume bullet points — stronger and ATS-friendly</li>
      <li>Personalised 30-day plan to get hired</li>
      <li>Ask any question about your resume</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])

    if uploaded:
        os.makedirs("temp", exist_ok=True)
        tmp_path = os.path.join("temp", uploaded.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner("Agent is analysing your resume — this takes 20-30 seconds on first run..."):
            try:
                from agent import CareerLensAgent
                agent   = CareerLensAgent()
                results = agent.run(tmp_path)

                st.session_state.agent    = agent
                st.session_state.results  = results
                st.session_state.analyzed = True

                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

                st.rerun()

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    st.stop()


# ── Load results ──────────────────────────────────────────────
results  = st.session_state.results
agent    = st.session_state.agent
summary  = results.get("summary", {})
gap      = results.get("gap", {})
jobs     = results.get("jobs", [])
rewrite  = results.get("rewrite", {})
plan     = results.get("plan", {})
parsed   = results.get("parsed", {})


# ══════════════════════════════════════════════════════════════
# PAGE 1 — ANALYSIS DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "Analysis Dashboard":
    name  = summary.get("name", "Candidate")
    score = summary.get("readiness_score", 0)
    color = "#22c55e" if score >= 70 else "#f59e0b" if score >= 45 else "#ef4444"

    st.markdown(f"<div class='pg-title'>Analysis Dashboard</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='pg-sub'>Resume analysis for {name}</div>", unsafe_allow_html=True)

    # ── Row 1: Key metrics ────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Readiness Score",    f"{score}/100")
    m2.metric("Skills Detected",    summary.get("skills_found", 0))
    m3.metric("Jobs Matched",       summary.get("jobs_found", 0))
    m4.metric("Verdict",            summary.get("verdict", ""))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Score gauge + skills ──────────────────────────
    gauge_col, skills_col = st.columns([1, 2])

    with gauge_col:
        fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw={"projection": "polar"})
        fig.patch.set_facecolor("#0f1729")
        ax.set_facecolor("#0f1729")

        theta_range = np.linspace(0, np.pi, 100)
        ax.plot(theta_range, [1]*100, color="#1a2540", linewidth=18, solid_capstyle="round")

        filled = int(score)
        theta_filled = np.linspace(0, np.pi * filled / 100, 100)
        bar_color = "#22c55e" if score >= 70 else "#f59e0b" if score >= 45 else "#ef4444"
        ax.plot(theta_filled, [1]*100, color=bar_color, linewidth=18, solid_capstyle="round")

        ax.set_ylim(0, 1.5)
        ax.set_theta_zero_location("W")
        ax.set_theta_direction(1)
        ax.axis("off")
        ax.text(0, -0.3, f"{score}", ha="center", va="center",
                fontsize=28, fontweight="bold", color="white",
                transform=ax.transData)
        ax.text(0, -0.65, "out of 100", ha="center", va="center",
                fontsize=9, color="#4a5568", transform=ax.transData)
        ax.text(0, -0.9, summary.get("verdict",""), ha="center", va="center",
                fontsize=9, color=bar_color, fontweight="bold", transform=ax.transData)
        st.pyplot(fig)
        plt.close()

    with skills_col:
        strengths = gap.get("strengths", [])
        missing   = gap.get("critical_missing", [])

        st.markdown("**Skills found in your resume**")
        if strengths:
            pills = " ".join([f"<span class='pill-green'>{s.title()}</span>" for s in strengths])
            st.markdown(pills, unsafe_allow_html=True)

        st.markdown("<br>**Critical skills missing**", unsafe_allow_html=True)
        if missing:
            pills = " ".join([f"<span class='pill-red'>{s.title()}</span>" for s in missing])
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#22c55e;font-size:0.88rem'>No critical gaps found.</p>", unsafe_allow_html=True)

        good = gap.get("good_to_have", [])
        if good:
            st.markdown("<br>**Good to have**", unsafe_allow_html=True)
            pills = " ".join([f"<span class='pill-yellow'>{s.title()}</span>" for s in good])
            st.markdown(pills, unsafe_allow_html=True)

    # ── Shortlist probability bar ─────────────────────────────
    st.markdown("---")
    shortlist_p = gap.get("shortlist_prob", 0)
    st.markdown(f"**Shortlisting Probability: {shortlist_p}%**")

    fig2, ax2 = plt.subplots(figsize=(8, 0.8))
    fig2.patch.set_facecolor("#0a0f1e")
    ax2.set_facecolor("#0a0f1e")
    ax2.barh([""], [shortlist_p],   color=bar_color, height=0.5)
    ax2.barh([""], [100-shortlist_p], left=[shortlist_p], color="#1a2540", height=0.5)
    ax2.set_xlim(0, 100)
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_xticklabels(["0%","25%","50%","75%","100%"], color="#4a5568", fontsize=8)
    ax2.tick_params(left=False, labelleft=False)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    st.pyplot(fig2)
    plt.close()

    # ── Improvement tips ──────────────────────────────────────
    tips = gap.get("improvement_tips", [])
    if tips:
        st.markdown("---")
        st.markdown("**Improvement Tips**")
        for tip in tips:
            st.markdown(f"<div class='card-accent'><p style='color:#94a3b8;font-size:0.88rem;margin:0'>{tip}</p></div>", unsafe_allow_html=True)

    # ── Matched Jobs ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Top Matching Jobs for Your Profile**")
    st.markdown("<p style='color:#4a5568;font-size:0.82rem'>Based on semantic matching of your resume against 241 real ML/AI fresher listings.</p>", unsafe_allow_html=True)

    if jobs:
        for i, job in enumerate(jobs[:8], 1):
            score_j = job.get("match_score", 0)
            color_j = "#22c55e" if score_j >= 70 else "#f59e0b" if score_j >= 50 else "#94a3b8"
            jc1, jc2, jc3 = st.columns([3, 1, 1])
            with jc1:
                st.markdown(
                    f"<div class='card'>"
                    f"<p style='color:#e2e8f0;font-size:0.95rem;font-weight:600;margin:0'>{job.get('role','')}</p>"
                    f"<p style='color:#4a5568;font-size:0.82rem;margin:4px 0 6px 0'>{job.get('company','')} — {job.get('type','')}</p>"
                    f"<p style='color:#64748b;font-size:0.78rem;margin:0'>{job.get('skills_required','')[:120]}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with jc2:
                st.markdown(
                    f"<div style='text-align:center;padding:1.2rem 0'>"
                    f"<p style='color:{color_j};font-size:1.4rem;font-weight:800;margin:0'>{score_j:.0f}%</p>"
                    f"<p style='color:#4a5568;font-size:0.75rem;margin:0'>match</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with jc3:
                st.markdown(
                    f"<div style='padding:1.2rem 0'>"
                    f"<p style='color:#4a5568;font-size:0.78rem;margin:0'>{job.get('experience','Fresher')}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
    else:
        st.info("No jobs found. Check your FAISS index.")


# ══════════════════════════════════════════════════════════════
# PAGE 2 — RESUME IMPROVEMENT
# ══════════════════════════════════════════════════════════════
elif page == "Resume Improvement":
    st.markdown("<div class='pg-title'>Resume Improvement</div>", unsafe_allow_html=True)
    st.markdown("<div class='pg-sub'>AI-powered suggestions to make your resume stronger and more ATS-friendly.</div>", unsafe_allow_html=True)

    bullets  = rewrite.get("rewritten_bullets", [])
    skills_s = rewrite.get("skills_section", "")
    proj_tips= rewrite.get("project_descriptions", [])

    # ── Rewritten Bullets ─────────────────────────────────────
    st.markdown("### Bullet Point Improvements")
    st.markdown("<p style='color:#4a5568;font-size:0.85rem'>Red = original weak version. Green = stronger rewritten version.</p>", unsafe_allow_html=True)

    if bullets:
        for item in bullets:
            st.markdown(
                f"<div class='card'>"
                f"<div class='original-text'>Before: {item['original']}</div>"
                f"<div class='rewritten-text'>After:  {item['rewritten']}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.markdown("<div class='card-green'><p style='color:#86efac;font-size:0.88rem;margin:0'>Your bullet points are already strong. No major rewrites needed.</p></div>", unsafe_allow_html=True)

    # ── Skills Section ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Suggested Skills Section")
    st.markdown("<p style='color:#4a5568;font-size:0.85rem'>Copy this into your resume skills section.</p>", unsafe_allow_html=True)

    if skills_s:
        st.markdown(
            f"<div class='card'><pre style='color:#94a3b8;font-size:0.88rem;margin:0;white-space:pre-wrap'>{skills_s}</pre></div>",
            unsafe_allow_html=True
        )

    # ── Project Tips ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Project Description Tips")

    if proj_tips:
        for pt in proj_tips:
            st.markdown(f"<div class='card-accent'>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:#e2e8f0;font-size:0.92rem;font-weight:600;margin-bottom:8px'>{pt['project']}</p>", unsafe_allow_html=True)
            for suggestion in pt["suggestions"]:
                st.markdown(f"<p style='color:#64748b;font-size:0.85rem;margin:4px 0'>• {suggestion}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — 30 DAY ACTION PLAN
# ══════════════════════════════════════════════════════════════
elif page == "30 Day Action Plan":
    name = plan.get("candidate_name", "Candidate")
    st.markdown("<div class='pg-title'>30 Day Action Plan</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='pg-sub'>Personalised plan for {name} to maximise shortlisting chances.</div>", unsafe_allow_html=True)

    # ── Weekly plan ───────────────────────────────────────────
    weekly = plan.get("weekly_plan", [])
    colors = ["#3b82f6", "#22c55e", "#f59e0b", "#a78bfa"]

    for i, week in enumerate(weekly):
        col_color = colors[i % len(colors)]
        st.markdown(
            f"<div class='card' style='border-left:4px solid {col_color}'>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='color:{col_color};font-size:0.82rem;margin:0'>{week['week']}</p>"
            f"<p style='color:#e2e8f0;font-size:1rem;font-weight:700;margin:4px 0 10px 0'>{week['title']}</p>",
            unsafe_allow_html=True
        )
        for task in week["tasks"]:
            st.markdown(
                f"<p class='task-item'>• {task}</p>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Learning Resources ────────────────────────────────────
    resources = plan.get("resources", [])
    if resources:
        st.markdown("---")
        st.markdown("### Learning Resources")
        for r in resources:
            st.markdown(
                f"<div class='card-accent'>"
                f"<p style='color:#3b82f6;font-size:0.8rem;margin:0'>{r['skill']} — {r['duration']}</p>"
                f"<p style='color:#94a3b8;font-size:0.88rem;margin:4px 0 0 0'>{r['resource']}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

    # ── Project Idea ──────────────────────────────────────────
    project_idea = plan.get("project_idea", "")
    if project_idea:
        st.markdown("---")
        st.markdown("### Recommended Next Project")
        st.markdown(
            f"<div class='card-green'>"
            f"<p style='color:#86efac;font-size:0.9rem;margin:0'>{project_idea}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

    # ── Target Companies ──────────────────────────────────────
    companies = plan.get("target_companies", [])
    if companies:
        st.markdown("---")
        st.markdown("### Companies to Target")
        st.markdown("<p style='color:#4a5568;font-size:0.82rem'>Search these on LinkedIn and Naukri with filter: 0-1 years, Bangalore.</p>", unsafe_allow_html=True)
        pills = " ".join([f"<span class='pill-blue'>{c}</span>" for c in companies])
        st.markdown(pills, unsafe_allow_html=True)

    # ── Interview Tips ────────────────────────────────────────
    itips = plan.get("interview_tips", [])
    if itips:
        st.markdown("---")
        st.markdown("### Interview Preparation Tips")
        for tip in itips:
            st.markdown(
                f"<div class='card-yellow'>"
                f"<p style='color:#fcd34d;font-size:0.78rem;margin:0'>{tip['skill']}</p>"
                f"<p style='color:#94a3b8;font-size:0.87rem;margin:6px 0 0 0'>{tip['tip']}</p>"
                f"</div>",
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════
# PAGE 4 — RESUME Q&A
# ══════════════════════════════════════════════════════════════
elif page == "Resume Q&A":
    st.markdown("<div class='pg-title'>Resume Q&A</div>", unsafe_allow_html=True)
    st.markdown("<div class='pg-sub'>Click a question below or type your own. Answers are retrieved instantly from the resume.</div>", unsafe_allow_html=True)

    # ── Store active question in session state ────────────────
    if "qa_question" not in st.session_state:
        st.session_state.qa_question = ""
    if "qa_answer" not in st.session_state:
        st.session_state.qa_answer = ""

    # ── Quick question buttons ────────────────────────────────
    st.markdown("**Click a question to get an instant answer:**")

    presets = [
        ("Technical Skills",      "List all the technical skills mentioned in this resume."),
        ("Programming Languages", "Which programming languages does this candidate know?"),
        ("ML & AI Knowledge",     "What machine learning and AI skills does this candidate have?"),
        ("Projects Built",        "Describe all the projects this candidate has built."),
        ("Tools & Frameworks",    "What tools, libraries and frameworks has this candidate used?"),
        ("Education",             "What is the educational background of this candidate?"),
        ("Work Experience",       "What work experience or training has this candidate completed?"),
        ("Certifications",        "What certifications has this candidate earned?"),
        ("Gen AI Skills",         "Does this candidate have any Generative AI or LLM related skills?"),
        ("Strongest Skills",      "What are the strongest and most highlighted skills in this resume?"),
        ("GitHub Projects",       "What GitHub projects has this candidate built and what do they do?"),
        ("Internship Details",    "Has this candidate done any internship or industry training?"),
    ]

    # Render in 3 columns
    qcols = st.columns(3)
    for i, (label, full_q) in enumerate(presets):
        col = qcols[i % 3]
        if col.button(label, use_container_width=True):
            st.session_state.qa_question = full_q
            with st.spinner("Searching resume..."):
                st.session_state.qa_answer = agent.answer_question(full_q)

    # ── Custom question ───────────────────────────────────────
    st.markdown("<br>**Or type your own question:**", unsafe_allow_html=True)

    custom_q = st.text_input(
        "Your question:",
        placeholder="e.g. Does this candidate have experience with LangChain and RAG?",
        label_visibility="collapsed",
    )

    if st.button("Get Answer", use_container_width=True):
        if custom_q.strip():
            st.session_state.qa_question = custom_q
            with st.spinner("Searching resume..."):
                st.session_state.qa_answer = agent.answer_question(custom_q)
        else:
            st.warning("Please type a question first.")

    # ── Show answer ───────────────────────────────────────────
    if st.session_state.qa_answer:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='color:#4a5568;font-size:0.8rem;margin-bottom:6px'>Question: {st.session_state.qa_question}</p>",
            unsafe_allow_html=True
        )
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        for chunk in st.session_state.qa_answer.split("---"):
            if chunk.strip():
                st.markdown(
                    f"<p style='color:#94a3b8;border-left:3px solid #2d3f6b;"
                    f"padding-left:12px;font-size:0.88rem;line-height:1.8;margin-bottom:12px'>"
                    f"{chunk.strip()}</p>",
                    unsafe_allow_html=True
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Resume summary ────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Auto-Generated Resume Summary**")
    st.markdown("<p style='color:#4a5568;font-size:0.82rem'>A concise summary of the most important information from the resume.</p>", unsafe_allow_html=True)

    if st.button("Generate Summary", use_container_width=True):
        with st.spinner("Summarising resume..."):
            summary_text = agent.get_summary()
        st.session_state.qa_answer   = ""   # clear Q&A when summary shown
        st.markdown(
            f"<div class='card'><p style='color:#cbd5e1;font-size:0.9rem;line-height:1.9'>{summary_text}</p></div>",
            unsafe_allow_html=True
        )
