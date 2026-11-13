"""
Semantic Grader — Streamlit Prototype
Modern Dashboard UI (Fully Responsive)
"""

import io
import os
import sys

# 1. Tell Python where to find the local 'grading' folder
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import streamlit as st
from datetime import datetime

# 2. Download required NLTK data for the cloud environment
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from grading.similarity import semantic_similarity_score, rubric_breakdown

st.set_page_config(
    page_title="Semantic Grader",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="auto"
)

# Mobile viewport meta
st.markdown(
    '<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">',
    unsafe_allow_html=True
)

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Session state
if "rubric_criteria" not in st.session_state:
    st.session_state.rubric_criteria = [
        {"keywords": "sunlight, water, CO2", "weight": 40},
        {"keywords": "glucose, oxygen", "weight": 30},
    ]
if "page" not in st.session_state:
    st.session_state.page = "Individual Grading"
if "grade_result" not in st.session_state:
    st.session_state.grade_result = None

# ==================== SIDEBAR ====================
PAGES = ["Individual Grading", "Batch Grading", "Rubric Builder", "How It Works", "Settings"]

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon">🎓</div>
        <div>
            <div class="logo-title">Semantic Grader</div>
            <div class="logo-sub">AI-Powered Short Answer Assessment</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # Gamitin ang st.radio para sa navigation. 
    # Ito ay gagana sa desktop at mobile nang walang aberya.
    selected_page = st.radio(
        "Navigation",
        PAGES,
        index=PAGES.index(st.session_state.page) if st.session_state.page in PAGES else 0,
        label_visibility="collapsed",
        key="main_navigation"
    )
    
    # Kapag nagbago ang pinindot, i-update ang session state at i-rerun
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()
    
    st.markdown("""
    <div class="sidebar-footer">
        <div class="footer-title">Better Feedback<br>Smarter Learning</div>
    </div>
    """, unsafe_allow_html=True)
# ==================== HEADER ====================
st.markdown("""
<div class="top-header">
    <div class="header-left">
        <h1>Semantic Grader</h1>
        <p class="header-sub">AI-Powered Short Answer Assessment</p>
    </div>
    <div class="header-right">
        <div class="status-badge">
            <span class="status-dot"></span>
            <div>
                <div class="status-title">Model Ready</div>
                <div class="status-sub">SBERT + Cosine Similarity</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== MOBILE TABS NAVIGATION ====================
# Lalabas lang ito sa mobile dahil sa CSS
st.markdown('<div class="mobile-tabs-container">', unsafe_allow_html=True)
mobile_page = st.radio(
    "Navigation",
    PAGES,
    index=PAGES.index(st.session_state.page) if st.session_state.page in PAGES else 0,
    horizontal=True,
    label_visibility="collapsed",
    key="mobile_tabs"
)
if mobile_page != st.session_state.page:
    st.session_state.page = mobile_page
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ==================== INDIVIDUAL GRADING ====================
if st.session_state.page == "Individual Grading":
    left_col, right_col = st.columns([1.15, 1], gap="large")
    
    with left_col:
        st.markdown("""
        <div class="panel-header">
            <span class="panel-icon">📝</span>
            <div>
                <div class="panel-title">Individual Grading</div>
                <div class="panel-sub">Enter the question, reference answer, and student response to get a semantic grade using SBERT and cosine similarity.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        question = st.text_input("Question", value="Explain the process of photosynthesis.", key="q_input")
        st.caption(f"{len(question)} / 500")
        
        reference = st.text_area(
            "Reference Answer",
            value="Photosynthesis is the process where plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.",
            height=100, key="ref_input"
        )
        st.caption(f"{len(reference)} / 500")
        
        student = st.text_area(
            "Student Response",
            value="Plants take in sunlight, CO2, and water to make their own food and release oxygen in the process.",
            height=100, key="stu_input"
        )
        st.caption(f"{len(student)} / 500")
        
        use_rubric = st.checkbox("Include optional rubric breakdown", value=True)
        
        btn_col1, btn_col2 = st.columns([1, 1.5])
        with btn_col1:
            if st.button("🔄  Clear", use_container_width=True, key="clear_btn"):
                st.session_state.grade_result = None
                st.rerun()
        with btn_col2:
            grade_btn = st.button("✨  Grade Response", type="primary", use_container_width=True, key="grade_btn")
        
        if grade_btn:
            with st.spinner("Encoding responses with SBERT..."):
                score = semantic_similarity_score(reference, student)
                st.session_state.grade_result = {
                    "score": score,
                    "question": question,
                    "reference": reference,
                    "student": student,
                    "timestamp": datetime.now().strftime("%b %d, %Y · %I:%M %p"),
                }
    
    with right_col:
        if st.session_state.grade_result is None:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">✨</div>
                <div class="empty-title">Grading Result</div>
                <div class="empty-sub">Enter a student response and click <b>Grade Response</b> to see the result here.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            r = st.session_state.grade_result
            score = r["score"]
            
            if score >= 80:
                verdict_title = "Excellent Match"
                verdict_desc = "The student's response is highly similar to the reference answer based on semantic meaning."
                verdict_class = "excellent"
            elif score >= 60:
                verdict_title = "Partial Match"
                verdict_desc = "Some key ideas are present, but part of the meaning diverges from the reference."
                verdict_class = "partial"
            else:
                verdict_title = "Weak Match"
                verdict_desc = "The response's meaning differs substantially from the reference answer."
                verdict_class = "weak"
            
            cosine = round(score / 100, 2)
            confidence = "High" if score >= 80 else ("Medium" if score >= 60 else "Low")
            score_pct = int(round(score))
            circumference = 2 * 3.14159 * 54
            dash = (score_pct / 100) * circumference
            
            st.markdown(f"""
            <div class="result-card">
                <div class="result-header">
                    <div class="result-header-left">
                        <span class="result-icon">✨</span>
                        <span class="result-title">Grading Result</span>
                    </div>
                    <div class="result-timestamp">{r['timestamp']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col_a, col_b = st.columns([1, 1.5])
            with col_a:
                st.markdown(f"""
                <div class="circular-score">
                    <svg width="140" height="140" viewBox="0 0 140 140">
                        <circle cx="70" cy="70" r="54" stroke="rgba(160,60,255,0.15)" stroke-width="10" fill="none"/>
                        <circle cx="70" cy="70" r="54" stroke="url(#grad)" stroke-width="10" fill="none"
                                stroke-linecap="round" stroke-dasharray="{dash} {circumference}"
                                transform="rotate(-90 70 70)"/>
                        <defs>
                            <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stop-color="#a855f7"/>
                                <stop offset="100%" stop-color="#7c3aed"/>
                            </linearGradient>
                        </defs>
                        <text x="70" y="68" text-anchor="middle" fill="#e5e7eb" font-size="28" font-weight="700">{score_pct}%</text>
                        <text x="70" y="88" text-anchor="middle" fill="#9ca3af" font-size="10">Semantic Score</text>
                    </svg>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div class="verdict-box {verdict_class}">
                    <div class="verdict-title">{verdict_title}</div>
                    <div class="verdict-desc">{verdict_desc}</div>
                    <div class="verdict-badge">Cosine Similarity · {cosine}</div>
                </div>
                """, unsafe_allow_html=True)
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="mini-metric">
                    <div class="mini-icon">🎯</div>
                    <div>
                        <div class="mini-label">Semantic Score</div>
                        <div class="mini-value">{score_pct}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="mini-metric">
                    <div class="mini-icon">📐</div>
                    <div>
                        <div class="mini-label">Cosine Similarity</div>
                        <div class="mini-value">{cosine}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="mini-metric">
                    <div class="mini-icon">✅</div>
                    <div>
                        <div class="mini-label">Confidence</div>
                        <div class="mini-value">{confidence}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="strengths-header">
                <span class="strengths-icon">💡</span>
                <span class="strengths-title">Key Strengths</span>
            </div>
            """, unsafe_allow_html=True)
            
            strengths = []
            if "sunlight" in student.lower() or "light" in student.lower():
                strengths.append("Mentions sunlight as an energy source")
            if "co2" in student.lower() or "carbon dioxide" in student.lower():
                strengths.append("Includes CO2 and water as inputs")
            if "oxygen" in student.lower() or "o2" in student.lower():
                strengths.append("Mentions oxygen production")
            if "food" in student.lower() or "glucose" in student.lower():
                strengths.append("Clearly explains the overall process")
            if not strengths:
                strengths = ["Response captures some semantic meaning"]
            
            strengths_html = "".join([f'<div class="strength-item">✅ {s}</div>' for s in strengths])
            st.markdown(f'<div class="strengths-list">{strengths_html}</div>', unsafe_allow_html=True)
            
            if use_rubric and st.session_state.rubric_criteria:
                with st.expander("📊  Detailed Comparison (Optional)"):
                    breakdown = rubric_breakdown(student, st.session_state.rubric_criteria)
                    df = pd.DataFrame([
                        {"Criterion": b["keywords"], "Status": "Present" if b["hit"] else "Not found", "Weight": f"{b['weight']}%"}
                        for b in breakdown
                    ])
                    st.dataframe(df, use_container_width=True, hide_index=True)
            
            a1, a2, a3 = st.columns(3)
            with a1:
                if st.button("📋  View Rubric", use_container_width=True, key="view_rubric"):
                    st.session_state.page = "Rubric Builder"
                    st.rerun()
            with a2:
                if st.button("🔄  Grade Another", use_container_width=True, key="grade_another"):
                    st.session_state.grade_result = None
                    st.rerun()
            with a3:
                export_data = f"Score,{score}\nCosine,{cosine}\nConfidence,{confidence}\n"
                st.download_button(
                    "📥  Export Result", data=export_data,
                    file_name=f"grading_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv", use_container_width=True, key="export_btn"
                )

# ==================== BATCH GRADING ====================
elif st.session_state.page == "Batch Grading":
    st.markdown("""
    <div class="panel-header">
        <span class="panel-icon">📊</span>
        <div>
            <div class="panel-title">Batch Grading</div>
            <div class="panel-sub">Upload a Google Forms CSV. Each question column is graded against its own reference answer.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CSV (Google Forms export supported)",
        type=["csv"],
    )

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)

        # Columns that are NOT questions
        meta_candidates = {"timestamp", "username", "email", "name",
                           "student_id", "id", "score", "submitted at"}
        question_cols = [
            c for c in raw_df.columns
            if c.strip().lower() not in meta_candidates
        ]

        if not question_cols:
            st.error("No question columns detected in the CSV.")
            st.stop()

        st.markdown("### 1️⃣  Map each question to a reference answer")
        st.caption(
            "Every non-meta column is treated as a question. "
            "Enter the reference answer used to grade that question."
        )

        # Persist references across reruns
        if "batch_refs" not in st.session_state:
            st.session_state.batch_refs = {
                "what is land?": "Land is the solid part of Earth's surface, not covered by water.",
                "what is ocean?": "An ocean is a large body of salt water that covers most of Earth's surface.",
            }

        references = {}
        for q in question_cols:
            references[q] = st.text_area(
                f"Reference for: **{q}**",
                value=st.session_state.batch_refs.get(q, ""),
                height=70,
                key=f"ref_{q}",
            )
        st.session_state.batch_refs = references

        # Identify student id column
        meta_lower = {c.lower().strip(): c for c in raw_df.columns}
        id_col = (
            meta_lower.get("student_id")
            or meta_lower.get("username")
            or meta_lower.get("email")
            or meta_lower.get("name")
            or raw_df.columns[0]
        )

        if st.button("✨  Process Batch", type="primary"):
            long_rows = []
            with st.spinner(f"Grading {len(raw_df)} students × {len(question_cols)} questions..."):
                for _, row in raw_df.iterrows():
                    for q in question_cols:
                        ref = references.get(q, "").strip()
                        ans = row[q]
                        if pd.isna(ans) or not ref:
                            continue
                        score = semantic_similarity_score(ref, str(ans))
                        long_rows.append({
                            "student_id": row[id_col],
                            "question":   q,
                            "response":   ans,
                            "reference_answer": ref,
                            "similarity_score": score,
                        })
            st.session_state["batch_result"] = pd.DataFrame(long_rows)

    if "batch_result" in st.session_state:
        result_df = st.session_state["batch_result"]

        # Per-question summary
        st.markdown("### 2️⃣  Results")
        summary = (
            result_df.groupby("question")["similarity_score"]
            .agg(n="count", avg="mean", min="min", max="max")
            .round(2).reset_index()
        )
        st.markdown("**Per-Question Summary**")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        # Optional wide view: one row per student, one column per question
        st.markdown("**Wide View (per student)**")
        wide = result_df.pivot_table(
            index="student_id", columns="question",
            values="similarity_score", aggfunc="mean"
        ).round(2).reset_index()
        st.dataframe(wide, use_container_width=True, hide_index=True)

        # Detailed long view
        with st.expander("📋  Detailed Results (long format)"):
            st.dataframe(result_df, use_container_width=True, hide_index=True)

        st.write(f"Overall class average: **{result_df['similarity_score'].mean():.1f}%**")

        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "📥  Export Results",
            data=csv_buffer.getvalue(),
            file_name="grading_results.csv",
            mime="text/csv",
        )

# ==================== RUBRIC BUILDER ====================
elif st.session_state.page == "Rubric Builder":
    st.markdown("""
    <div class="panel-header">
        <span class="panel-icon">📋</span>
        <div>
            <div class="panel-title">Rubric Builder</div>
            <div class="panel-sub">Define keyword-based criteria for instructor feedback. Does not affect the final grade.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    to_remove = None
    for i, crit in enumerate(st.session_state.rubric_criteria):
        c1, c2, c3 = st.columns([3, 1, 0.5])
        with c1:
            crit["keywords"] = st.text_input(f"Keywords {i + 1}", value=crit["keywords"], key=f"kw_{i}")
        with c2:
            crit["weight"] = st.number_input(f"Weight % {i + 1}", min_value=0, max_value=100, value=crit["weight"], key=f"wt_{i}")
        with c3:
            st.write("")
            if st.button("❌", key=f"rm_{i}"):
                to_remove = i
    if to_remove is not None:
        st.session_state.rubric_criteria.pop(to_remove)
        st.rerun()
    if st.button("➕  Add Criterion"):
        st.session_state.rubric_criteria.append({"keywords": "", "weight": 0})
        st.rerun()
    total_weight = sum(c["weight"] for c in st.session_state.rubric_criteria)
    if total_weight == 100:
        st.success(f"Total weight: {total_weight}%")
    else:
        st.warning(f"Total weight: {total_weight}% (should sum to 100%)")

# ==================== HOW IT WORKS ====================
elif st.session_state.page == "How It Works":
    st.markdown("""
    <div class="panel-header">
        <span class="panel-icon">❓</span>
        <div>
            <div class="panel-title">How It Works</div>
            <div class="panel-sub">Understanding the semantic grading process.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    ### The Process
    1. **Text Preprocessing** — The reference answer and student response are normalized.
    2. **SBERT Encoding** — Both texts are encoded into dense semantic vectors using Sentence-BERT.
    3. **Cosine Similarity** — The angle between the two vectors is computed.
    4. **Final Score** — The cosine similarity is converted into a 0–100 scale.
    
    ### Why Semantic?
    Unlike keyword matching, semantic grading understands *meaning*.
    """)

# ==================== SETTINGS ====================
elif st.session_state.page == "Settings":
    st.markdown("""
    <div class="panel-header">
        <span class="panel-icon">⚙️</span>
        <div>
            <div class="panel-title">Settings</div>
            <div class="panel-sub">Configure the grading engine.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.text_input("SBERT Model", value="all-MiniLM-L6-v2", disabled=True)
    st.slider("Threshold — Excellent Match", 0, 100, 80, disabled=True)
    st.slider("Threshold — Partial Match", 0, 100, 60, disabled=True)
    st.info("Settings are read-only in this prototype.")
