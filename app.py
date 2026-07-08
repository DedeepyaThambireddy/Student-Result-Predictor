import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Student Result Predictor",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  THEME — warm teal/emerald, student-friendly
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* background */
.stApp {
    background: linear-gradient(135deg, #0b1f1a 0%, #0d2b24 50%, #0a1f1c 100%);
    min-height: 100vh;
}
header[data-testid="stHeader"] { background: transparent; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071a16 0%, #0d2b24 100%) !important;
    border-right: 1px solid rgba(52,211,153,0.15);
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }

.sidebar-brand {
    background: linear-gradient(135deg, rgba(52,211,153,0.12), rgba(16,185,129,0.08));
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 16px;
    padding: 22px 16px;
    text-align: center;
    margin-bottom: 28px;
}
.sidebar-brand .sb-icon { font-size: 2.4rem; }
.sidebar-brand h2 {
    color: #34d399 !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    margin: 8px 0 4px 0 !important;
    letter-spacing: -0.3px;
}
.sidebar-brand p {
    color: rgba(255,255,255,0.4) !important;
    font-size: 0.7rem !important;
    margin: 0 !important;
}

/* nav buttons */
.nav-btn {
    display: flex;
    align-items: center;
    gap: 12px;
    width: 100%;
    padding: 12px 16px;
    border-radius: 10px;
    border: 1px solid transparent;
    background: transparent;
    color: rgba(255,255,255,0.55) !important;
    font-size: 0.9rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.18s;
    margin-bottom: 4px;
    text-decoration: none;
}
.nav-btn:hover {
    background: rgba(52,211,153,0.08);
    border-color: rgba(52,211,153,0.2);
    color: #34d399 !important;
}
.nav-btn.active {
    background: rgba(52,211,153,0.15);
    border-color: rgba(52,211,153,0.35);
    color: #34d399 !important;
    font-weight: 600;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0d2b24 0%, #0f3328 60%, #0b2820 100%);
    border: 1px solid rgba(52,211,153,0.2);
    border-radius: 22px;
    padding: 48px 52px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(52,211,153,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    display: inline-block;
    background: rgba(52,211,153,0.12);
    border: 1px solid rgba(52,211,153,0.3);
    color: #34d399;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 18px;
}
.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 10px 0;
    line-height: 1.15;
    letter-spacing: -0.8px;
}
.hero-title span { color: #34d399; }
.hero-sub {
    color: rgba(255,255,255,0.5);
    font-size: 1rem;
    line-height: 1.7;
    max-width: 640px;
    margin: 0;
}

/* ── Home cards (how-to) ── */
.how-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(52,211,153,0.12);
    border-radius: 14px;
    padding: 22px 20px;
    height: 100%;
    transition: border-color 0.2s;
}
.how-card:hover { border-color: rgba(52,211,153,0.3); }
.how-card .step-num {
    background: rgba(52,211,153,0.15);
    color: #34d399;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    padding: 3px 10px;
    border-radius: 20px;
    display: inline-block;
    margin-bottom: 12px;
    text-transform: uppercase;
}
.how-card h4 {
    color: rgba(255,255,255,0.9);
    font-size: 0.95rem;
    font-weight: 600;
    margin: 0 0 8px 0;
}
.how-card p {
    color: rgba(255,255,255,0.45);
    font-size: 0.82rem;
    line-height: 1.65;
    margin: 0;
}

/* ── Stat cards ── */
.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: rgba(52,211,153,0.3); }
.stat-card .val { font-size: 1.9rem; font-weight: 700; color: #34d399; line-height: 1; }
.stat-card .lbl { font-size: 0.72rem; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 1px; margin-top: 6px; }

/* ── Section titles ── */
.sec-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: rgba(255,255,255,0.88);
    border-left: 3px solid #34d399;
    padding-left: 12px;
    margin: 28px 0 14px 0;
}

/* ── Result banners ── */
.res-pass {
    background: linear-gradient(135deg, rgba(52,211,153,0.18), rgba(52,211,153,0.06));
    border: 2px solid rgba(52,211,153,0.55);
    border-radius: 18px;
    padding: 36px 40px;
    text-align: center;
    margin: 24px 0;
    animation: popIn 0.45s ease;
}
.res-fail {
    background: linear-gradient(135deg, rgba(245,101,101,0.18), rgba(245,101,101,0.06));
    border: 2px solid rgba(245,101,101,0.55);
    border-radius: 18px;
    padding: 36px 40px;
    text-align: center;
    margin: 24px 0;
    animation: popIn 0.45s ease;
}
@keyframes popIn {
    from { opacity: 0; transform: scale(0.95) translateY(12px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
}
.res-emoji  { font-size: 3rem; display: block; margin-bottom: 10px; }
.res-label  { font-size: 2.6rem; font-weight: 800; letter-spacing: 4px; display: block; margin-bottom: 6px; }
.label-pass { color: #34d399; }
.label-fail { color: #f56565; }
.res-conf   { font-size: 0.95rem; color: rgba(255,255,255,0.55); }
.res-conf b { color: white; font-size: 1.1rem; }

/* ── Tip box (after fail) ── */
.tip-box {
    background: rgba(251,191,36,0.07);
    border: 1px solid rgba(251,191,36,0.25);
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 16px;
}
.tip-box .tip-title { color: #fbbf24; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
.tip-box ul { color: rgba(255,255,255,0.65); font-size: 0.83rem; line-height: 1.9; margin: 0; padding-left: 16px; }

/* ── Input summary table ── */
.inp-table { width: 100%; border-collapse: collapse; border-radius: 12px; overflow: hidden; }
.inp-table th { background: rgba(52,211,153,0.1); color: #34d399; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; padding: 10px 14px; text-align: left; }
.inp-table td { color: rgba(255,255,255,0.75); font-size: 0.85rem; padding: 9px 14px; border-top: 1px solid rgba(255,255,255,0.05); }

/* ── Metric cards ── */
.mc { border-radius: 14px; padding: 20px 16px; text-align: center; border: 1px solid; }
.mc .mv { font-size: 2rem; font-weight: 700; line-height: 1; }
.mc .ml { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px; font-weight: 600; }
.mc .md { font-size: 0.68rem; margin-top: 7px; opacity: 0.6; }
.mc-teal   { background: rgba(52,211,153,0.08);  border-color: rgba(52,211,153,0.3);  }
.mc-teal   .mv, .mc-teal   .ml { color: #34d399; }
.mc-green  { background: rgba(16,185,129,0.08);  border-color: rgba(16,185,129,0.3);  }
.mc-green  .mv, .mc-green  .ml { color: #10b981; }
.mc-purple { background: rgba(139,92,246,0.08);  border-color: rgba(139,92,246,0.3);  }
.mc-purple .mv, .mc-purple .ml { color: #8b5cf6; }
.mc-amber  { background: rgba(251,191,36,0.08);  border-color: rgba(251,191,36,0.3);  }
.mc-amber  .mv, .mc-amber  .ml { color: #fbbf24; }

/* sliders teal */
.stSlider [data-baseweb="slider"] [role="slider"] { background: #34d399 !important; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #059669, #34d399);
    color: #071a16 !important;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 10px;
    padding: 12px 40px;
    width: 100%;
    transition: opacity 0.18s, transform 0.1s;
    letter-spacing: 0.3px;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

/* ── Footer ── */
.footer {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 16px 24px;
    text-align: center;
    margin-top: 48px;
    color: rgba(255,255,255,0.25) !important;
    font-size: 0.75rem;
}

label { color: rgba(255,255,255,0.75) !important; }
p, span { color: rgba(255,255,255,0.75); }
.streamlit-expanderHeader { color: rgba(255,255,255,0.7) !important; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODEL + DATA  (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_everything():
    model  = joblib.load("student_model.pkl")
    scaler = joblib.load("scaler.pkl")
    df     = pd.read_excel("student_dataset.xlsx")

    # rebuild test split identically to evaluate_model.py
    X = df.drop("Result", axis=1)
    y = df["Result"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_sc = scaler.transform(X_test)
    y_pred    = model.predict(X_test_sc)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred)  * 100, 1),
        "precision": round(precision_score(y_test, y_pred) * 100, 1),
        "recall":    round(recall_score(y_test, y_pred)    * 100, 1),
        "f1":        round(f1_score(y_test, y_pred)        * 100, 1),
        "cm":        confusion_matrix(y_test, y_pred),
        "coef":      model.coef_[0],
    }
    return model, scaler, df, metrics

model, scaler, df, metrics = load_everything()

total      = len(df)
pass_count = int(df["Result"].sum())
fail_count = total - pass_count
pass_rate  = round(pass_count / total * 100, 1)

FEATURES = ["Study_Hours", "Attendance", "Previous_Marks", "Assignments", "Sleep_Hours"]

# ─────────────────────────────────────────────
#  SIDEBAR  — navigation
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sb-icon">📖</div>
        <h2>Student Result Prediction System</h2>
        <p>Will I pass?</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.7rem; color:rgba(255,255,255,0.3); text-transform:uppercase; letter-spacing:1.5px; margin-bottom:8px;'>Navigation</div>", unsafe_allow_html=True)

    pages = {
        "🏠  Home":             "Home",
        "🔮  Predict My Result": "Predict",
        "📊  Data Analysis":    "Analysis",
        "📈  Model Performance": "Performance",
    }

    if "page" not in st.session_state:
        st.session_state.page = "Home"

    for label, key in pages.items():
        active = "active" if st.session_state.page == key else ""
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.78rem; color:rgba(255,255,255,0.35); line-height:1.9;">
    📂 Total students &nbsp;<b style="color:#34d399">{total}</b><br>
    ✅ Passed &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b style="color:#34d399">{pass_count}</b><br>
    ❌ Failed &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b style="color:#f56565">{fail_count}</b><br>
    🎯 Pass rate &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b style="color:#34d399">{pass_rate}%</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.7rem; color:rgba(255,255,255,0.25); text-align:center; line-height:1.8;">
    Developed by<br>
    <span style="color:#34d399; font-weight:600;">Dedeepya Thambireddy</span><br>
    </div>
    """, unsafe_allow_html=True)

page = st.session_state.page


# ══════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════
if page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">Student Self-Check Tool</div>
        <h1 class="hero-title">Find out if you're on track<br>to <span>pass your exams 🎓</span></h1>
        <p class="hero-sub">
            Just enter a few details about your study habits and academic record —
            this tool will instantly tell you whether you're likely to pass or need
            to put in a bit more effort before your exams.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats
    q1, q2, q3, q4 = st.columns(4)
    for col, val, lbl in [
        (q1, str(total),       "Students in Dataset"),
        (q2, str(pass_count),  "Students Who Passed"),
        (q3, str(fail_count),  "Students Who Failed"),
        (q4, f"{pass_rate}%",  "Overall Pass Rate"),
    ]:
        with col:
            st.markdown(f'<div class="stat-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # How it works
    st.markdown('<div class="sec-title">How to use this tool</div>', unsafe_allow_html=True)
    h1, h2, h3, h4 = st.columns(4)
    cards = [
        ("Step 1", "📝  Enter your details",
         "Go to Predict My Result in the sidebar and fill in your study hours, attendance, previous marks, assignments done, and sleep hours."),
        ("Step 2", "⚡  Get your prediction",
         "Click the Predict button. The tool will instantly tell you whether you're likely to PASS or FAIL, with a confidence percentage."),
        ("Step 3", "💡  Read your tips",
         "If the result is FAIL, you'll see personalised tips on which areas to focus on to improve your chances before the exam."),
        ("Step 4", "📊  Explore the data",
         "Check the Data Analysis page to see how students like you perform, or visit Model Performance to understand how the tool works."),
    ]
    for col, (step, title, desc) in zip([h1, h2, h3, h4], cards):
        with col:
            st.markdown(f"""
            <div class="how-card">
                <div class="step-num">{step}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # What the tool looks at
    st.markdown('<div class="sec-title">What this tool looks at</div>', unsafe_allow_html=True)
    f1c, f2c, f3c, f4c, f5c = st.columns(5)
    factors = [
        (f1c, "📚", "Study Hours", "How many hours a day you study"),
        (f2c, "🏫", "Attendance",  "How regularly you attend class"),
        (f3c, "📝", "Previous Marks", "Your score in the last exam"),
        (f4c, "✅", "Assignments",  "Assignments you've submitted"),
        (f5c, "😴", "Sleep Hours",  "Your average sleep per night"),
    ]
    for col, icon, name, desc in factors:
        with col:
            st.markdown(f"""
            <div class="how-card" style="text-align:center;">
                <div style="font-size:2rem; margin-bottom:10px;">{icon}</div>
                <h4 style="text-align:center;">{name}</h4>
                <p style="text-align:center;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        📖 Student Result Prediction System &nbsp;·&nbsp;
        Developed by <b>Dedeepya Thambireddy</b>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE: PREDICT
# ══════════════════════════════════════════════
elif page == "Predict":

    st.markdown("""
    <div class="hero" style="padding:36px 44px;">
        <div class="hero-eyebrow">Step 2 of 2</div>
        <h1 class="hero-title" style="font-size:1.9rem;">Enter your details below 👇</h1>
        <p class="hero-sub" style="font-size:0.9rem;">
            Be as accurate as possible — the more honest your inputs, the more useful the result.
        </p>
    </div>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        st.markdown('<div class="sec-title">Your Academic Profile</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            study_hours  = st.slider("📚 Study Hours per day",   0.0, 12.0, 5.0, 0.5,
                                     help="Average hours you study daily outside class")
            prev_marks   = st.slider("📝 Previous Exam Marks",   0,   100,  60,
                                     help="Your score (out of 100) in the last exam")
            sleep_hours  = st.slider("😴 Sleep Hours per night", 4.0, 10.0, 7.0, 0.5,
                                     help="Average sleep you get each night")
        with b:
            attendance   = st.slider("🏫 Attendance (%)",        0,   100,  75,
                                     help="Percentage of classes you attend")
            assignments  = st.slider("✅ Assignments Submitted", 0,   10,   6,
                                     help="Number of assignments you've submitted")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡  Check My Result", use_container_width=True)

    with right_col:
        st.markdown('<div class="sec-title">Your Input Summary</div>', unsafe_allow_html=True)
        inp_rows = [
            ("📚 Study Hours",    f"{study_hours} hrs/day"),
            ("🏫 Attendance",     f"{attendance}%"),
            ("📝 Previous Marks", f"{prev_marks} / 100"),
            ("✅ Assignments",    f"{assignments} / 10"),
            ("😴 Sleep Hours",    f"{sleep_hours} hrs/night"),
        ]
        tbl = '<table class="inp-table"><tr><th>Factor</th><th>Your Value</th></tr>'
        for r, v in inp_rows:
            tbl += f"<tr><td>{r}</td><td>{v}</td></tr>"
        tbl += "</table>"
        st.markdown(tbl, unsafe_allow_html=True)

        # Quick readiness score (visual only)
        readiness = int(min(100, (
            (study_hours / 12) * 30 +
            (attendance  / 100) * 25 +
            (prev_marks  / 100) * 25 +
            (assignments / 10)  * 15 +
            (1 - abs(sleep_hours - 7.5) / 3.5) * 5
        )))
        color = "#34d399" if readiness >= 60 else "#fbbf24" if readiness >= 40 else "#f56565"
        st.markdown(f"""
        <div style="margin-top:18px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
                    border-radius:12px; padding:16px;">
            <div style="font-size:0.72rem; color:rgba(255,255,255,0.4); text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">
                Readiness Score
            </div>
            <div style="font-size:2rem; font-weight:800; color:{color};">{readiness}<span style="font-size:1rem; font-weight:400; color:rgba(255,255,255,0.3);"> / 100</span></div>
            <div style="font-size:0.75rem; color:rgba(255,255,255,0.35); margin-top:4px;">Based on your inputs</div>
        </div>
        """, unsafe_allow_html=True)

    # ── PREDICTION RESULT ──
    if predict_btn:
        inp_scaled = scaler.transform([[study_hours, attendance, prev_marks, assignments, sleep_hours]])
        pred  = model.predict(inp_scaled)[0]
        proba = model.predict_proba(inp_scaled)[0]
        conf  = proba[1] * 100 if pred == 1 else proba[0] * 100

        if pred == 1:
            st.markdown(f"""
            <div class="res-pass">
                <span class="res-emoji">🏆</span>
                <span class="res-label label-pass">YOU'RE LIKELY TO PASS!</span>
                <div class="res-conf">Confidence: <b>{conf:.1f}%</b> &nbsp;·&nbsp; Keep it up!</div>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
            <div class="res-fail">
                <span class="res-emoji">⚠️</span>
                <span class="res-label label-fail">AT RISK OF FAILING</span>
                <div class="res-conf">Confidence: <b>{conf:.1f}%</b> &nbsp;·&nbsp; But you still have time to turn it around!</div>
            </div>
            """, unsafe_allow_html=True)

            # Personalised tips based on weakest inputs
            tips = []
            if study_hours < 5:
                tips.append(f"📚 Try to study at least <b>5–6 hours a day</b> — you're currently at {study_hours} hrs.")
            if attendance < 75:
                tips.append(f"🏫 Raise your attendance above <b>75%</b> — you're at {attendance}% right now.")
            if prev_marks < 55:
                tips.append(f"📝 Review your weak subjects — your previous score of <b>{prev_marks}</b> needs improvement.")
            if assignments < 6:
                tips.append(f"✅ Complete more assignments — you've only done <b>{assignments}/10</b>.")
            if sleep_hours < 6 or sleep_hours > 9:
                tips.append(f"😴 Aim for <b>7–8 hours of sleep</b> — {sleep_hours} hrs is affecting your performance.")
            if not tips:
                tips.append("💡 Focus on consistent revision and practice papers in the days ahead.")

            tip_html = "".join(f"<li>{t}</li>" for t in tips)
            st.markdown(f"""
            <div class="tip-box">
                <div class="tip-title">💡 What you can do right now</div>
                <ul>{tip_html}</ul>
            </div>
            """, unsafe_allow_html=True)

        # ── Probability bars ──
        st.markdown('<div class="sec-title">Probability Breakdown</div>', unsafe_allow_html=True)
        pb1, pb2 = st.columns(2)
        with pb1:
            st.markdown("**✅ Pass Probability**")
            st.progress(int(proba[1] * 100))
            st.markdown(f"<span style='color:#34d399; font-weight:700; font-size:1.1rem;'>{proba[1]*100:.1f}%</span>", unsafe_allow_html=True)
        with pb2:
            st.markdown("**❌ Fail Probability**")
            st.progress(int(proba[0] * 100))
            st.markdown(f"<span style='color:#f56565; font-weight:700; font-size:1.1rem;'>{proba[0]*100:.1f}%</span>", unsafe_allow_html=True)

        # ── Radar chart: you vs average passer ──
        st.markdown('<div class="sec-title">Your Profile vs Average Student Who Passed</div>', unsafe_allow_html=True)

        pass_avg = df[df["Result"] == 1][FEATURES].mean()

        ranges = {
            "Study_Hours":    (0, 12),
            "Attendance":     (0, 100),
            "Previous_Marks": (0, 100),
            "Assignments":    (0, 10),
            "Sleep_Hours":    (4, 10),
        }
        radar_labels = ["Study Hours", "Attendance", "Prev Marks", "Assignments", "Sleep"]

        def norm(v, mn, mx): return (v - mn) / (mx - mn)

        you_vals  = [norm(v, *ranges[k]) for v, k in zip(
            [study_hours, attendance, prev_marks, assignments, sleep_hours], ranges)]
        pass_vals = [norm(pass_avg[k], *ranges[k]) for k in ranges]

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=you_vals + [you_vals[0]], theta=radar_labels + [radar_labels[0]],
            fill='toself', name='You',
            line_color='#34d399', fillcolor='rgba(52,211,153,0.2)',
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=pass_vals + [pass_vals[0]], theta=radar_labels + [radar_labels[0]],
            fill='toself', name='Avg Passer',
            line_color='#8b5cf6', fillcolor='rgba(139,92,246,0.12)',
        ))
        fig_r.update_layout(
            polar=dict(
                bgcolor='rgba(255,255,255,0.02)',
                radialaxis=dict(visible=True, range=[0,1], color='rgba(255,255,255,0.25)',
                                tickfont=dict(size=8, color='rgba(255,255,255,0.3)')),
                angularaxis=dict(color='rgba(255,255,255,0.4)',
                                 tickfont=dict(size=11, color='rgba(255,255,255,0.65)')),
            ),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(255,255,255,0.7)'),
            legend=dict(bgcolor='rgba(255,255,255,0.04)', borderwidth=0,
                        font=dict(color='rgba(255,255,255,0.65)')),
            height=380, margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("""
    <div class="footer">
        📖 Student Result Prediction System &nbsp;·&nbsp; Developed by <b>Dedeepya Thambireddy</b>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE: DATA ANALYSIS
# ══════════════════════════════════════════════
elif page == "Analysis":

    st.markdown("""
    <div class="hero" style="padding:32px 44px;">
        <div class="hero-eyebrow">Data Analysis</div>
        <h1 class="hero-title" style="font-size:1.9rem;">How do students in the dataset perform? 📊</h1>
        <p class="hero-sub" style="font-size:0.88rem;">
            Explore the patterns behind pass and fail outcomes across all students.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    d1, d2, d3, d4 = st.columns(4)
    for col, val, lbl in [
        (d1, str(total),      "Total Records"),
        (d2, str(pass_count), "Passed"),
        (d3, str(fail_count), "Failed"),
        (d4, f"{pass_rate}%", "Pass Rate"),
    ]:
        with col:
            st.markdown(f'<div class="stat-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    CHART_LAYOUT = dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
        font=dict(color='rgba(255,255,255,0.65)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.06)', color='rgba(255,255,255,0.45)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)', color='rgba(255,255,255,0.45)'),
        legend=dict(bgcolor='rgba(255,255,255,0.04)', borderwidth=0),
        margin=dict(t=20, b=20),
    )
    COLOR_MAP = {"Pass": "#34d399", "Fail": "#f56565"}
    df_plot = df.copy()
    df_plot["Outcome"] = df_plot["Result"].map({1: "Pass", 0: "Fail"})

    # Row 1 — scatter plots
    r1a, r1b = st.columns(2)
    with r1a:
        st.markdown('<div class="sec-title">Study Hours vs Previous Marks</div>', unsafe_allow_html=True)
        f1 = px.scatter(df_plot, x="Study_Hours", y="Previous_Marks", color="Outcome",
                        color_discrete_map=COLOR_MAP, opacity=0.65,
                        labels={"Study_Hours": "Study Hours/Day", "Previous_Marks": "Previous Marks"})
        f1.update_layout(**CHART_LAYOUT, height=310)
        st.plotly_chart(f1, use_container_width=True)

    with r1b:
        st.markdown('<div class="sec-title">Attendance vs Previous Marks</div>', unsafe_allow_html=True)
        f2 = px.scatter(df_plot, x="Attendance", y="Previous_Marks", color="Outcome",
                        color_discrete_map=COLOR_MAP, opacity=0.65,
                        labels={"Attendance": "Attendance (%)", "Previous_Marks": "Previous Marks"})
        f2.update_layout(**CHART_LAYOUT, height=310)
        st.plotly_chart(f2, use_container_width=True)

    # Row 2 — donut + histogram
    r2a, r2b = st.columns(2)
    with r2a:
        st.markdown('<div class="sec-title">Pass / Fail Split</div>', unsafe_allow_html=True)
        f3 = go.Figure(data=[go.Pie(
            labels=["Pass", "Fail"], values=[pass_count, fail_count],
            marker_colors=["#34d399", "#f56565"], hole=0.55,
            textinfo='label+percent', textfont=dict(size=13, color='white'),
        )])
        f3.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                         font=dict(color='rgba(255,255,255,0.65)'),
                         legend=dict(bgcolor='rgba(255,255,255,0.04)', borderwidth=0),
                         height=300, margin=dict(t=20, b=20),
                         annotations=[dict(text='Result', x=0.5, y=0.5, font_size=14,
                                           showarrow=False, font_color='rgba(255,255,255,0.5)')])
        st.plotly_chart(f3, use_container_width=True)

    with r2b:
        st.markdown('<div class="sec-title">Study Hours Distribution</div>', unsafe_allow_html=True)
        f4 = go.Figure()
        f4.add_trace(go.Histogram(x=df_plot[df_plot["Result"]==1]["Study_Hours"],
                                   name="Pass", marker_color="#34d399", opacity=0.7, nbinsx=20))
        f4.add_trace(go.Histogram(x=df_plot[df_plot["Result"]==0]["Study_Hours"],
                                   name="Fail", marker_color="#f56565", opacity=0.7, nbinsx=20))
        f4.update_layout(**CHART_LAYOUT, height=300, barmode='overlay',
                         xaxis_title="Study Hours/Day", yaxis_title="Count")
        st.plotly_chart(f4, use_container_width=True)

    # Row 3 — box plots
    r3a, r3b = st.columns(2)
    with r3a:
        st.markdown('<div class="sec-title">Attendance by Outcome</div>', unsafe_allow_html=True)
        f5 = px.box(df_plot, x="Outcome", y="Attendance", color="Outcome",
                    color_discrete_map=COLOR_MAP,
                    labels={"Outcome": "", "Attendance": "Attendance (%)"})
        f5.update_layout(**CHART_LAYOUT, height=300, showlegend=False)
        st.plotly_chart(f5, use_container_width=True)

    with r3b:
        st.markdown('<div class="sec-title">Assignments by Outcome</div>', unsafe_allow_html=True)
        f6 = px.box(df_plot, x="Outcome", y="Assignments", color="Outcome",
                    color_discrete_map=COLOR_MAP,
                    labels={"Outcome": "", "Assignments": "Assignments Submitted"})
        f6.update_layout(**CHART_LAYOUT, height=300, showlegend=False)
        st.plotly_chart(f6, use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="sec-title">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    corr = df[FEATURES + ["Result"]].corr()
    f7 = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=["Study Hrs", "Attendance", "Prev Marks", "Assignments", "Sleep", "Result"],
        y=["Study Hrs", "Attendance", "Prev Marks", "Assignments", "Sleep", "Result"],
        colorscale=[[0,"#f56565"],[0.5,"#0d2b24"],[1,"#34d399"]],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}", textfont=dict(size=10, color='white'),
        showscale=True,
    ))
    f7.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                     font=dict(color='rgba(255,255,255,0.65)'),
                     height=380, margin=dict(t=20, b=20),
                     xaxis=dict(tickfont=dict(size=10)),
                     yaxis=dict(tickfont=dict(size=10)))
    st.plotly_chart(f7, use_container_width=True)

    # Raw data
    with st.expander("🔍 Browse the raw dataset"):
        show_df = df.copy()
        show_df["Result"] = show_df["Result"].map({1: "✅ Pass", 0: "❌ Fail"})
        st.dataframe(show_df, use_container_width=True)

    st.markdown("""
    <div class="footer">
        📖 Student Result Prediction System &nbsp;·&nbsp; Developed by <b>Dedeepya Thambireddy</b>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════
elif page == "Performance":

    st.markdown("""
    <div class="hero" style="padding:32px 44px;">
        <div class="hero-eyebrow">Model Performance</div>
        <h1 class="hero-title" style="font-size:1.9rem;">How accurate is this tool? 📈</h1>
        <p class="hero-sub" style="font-size:0.88rem;">
            Here's an honest look at how well the prediction model performs on students it has never seen before.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    mdata = [
        (m1, "mc-teal",   f"{metrics['accuracy']}%",  "Accuracy",  "Correctly classified"),
        (m2, "mc-green",  f"{metrics['precision']}%", "Precision", "Right when it says PASS"),
        (m3, "mc-purple", f"{metrics['recall']}%",    "Recall",    "Pass students caught"),
        (m4, "mc-amber",  f"{metrics['f1']}%",        "F1 Score",  "Balanced reliability"),
    ]
    for col, cls, val, lbl, desc in mdata:
        with col:
            st.markdown(f"""
            <div class="mc {cls}">
                <div class="mv">{val}</div>
                <div class="ml">{lbl}</div>
                <div class="md">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Confusion matrix + feature importance
    cm_col, fi_col = st.columns(2)

    with cm_col:
        st.markdown('<div class="sec-title">Confusion Matrix</div>', unsafe_allow_html=True)
        cm_vals = metrics["cm"]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_vals,
            x=["Predicted Fail", "Predicted Pass"],
            y=["Actual Fail", "Actual Pass"],
            colorscale=[[0, "#0d2b24"], [1, "#34d399"]],
            text=[[str(v) for v in row] for row in cm_vals],
            texttemplate="<b>%{text}</b>",
            textfont=dict(size=26, color='white'),
            showscale=False,
        ))
        fig_cm.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(255,255,255,0.65)', size=12),
            xaxis=dict(side='bottom', tickfont=dict(size=11, color='rgba(255,255,255,0.55)')),
            yaxis=dict(tickfont=dict(size=11, color='rgba(255,255,255,0.55)')),
            height=320, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        tn, fp, fn, tp = cm_vals.ravel()
        st.markdown(f"""
        <div style="font-size:0.8rem; background:rgba(255,255,255,0.03); border-radius:10px; padding:12px 16px; line-height:2;">
        ✅ <b style="color:#34d399;">Correctly predicted PASS (TP):</b> {tp} &nbsp;|&nbsp;
        ❌ <b style="color:#f56565;">False alarm (FP):</b> {fp}<br>
        🔵 <b style="color:#8b5cf6;">Correctly predicted FAIL (TN):</b> {tn} &nbsp;|&nbsp;
        🟠 <b style="color:#fbbf24;">Missed at-risk student (FN):</b> {fn}
        </div>
        """, unsafe_allow_html=True)

    with fi_col:
        st.markdown('<div class="sec-title">Which factors matter most?</div>', unsafe_allow_html=True)
        feat_labels = ["Study Hours", "Attendance", "Prev Marks", "Assignments", "Sleep Hours"]
        coefs = np.abs(metrics["coef"])
        idx   = np.argsort(coefs)
        colors = ["#34d399", "#10b981", "#8b5cf6", "#fbbf24", "#f59e0b"]

        fig_fi = go.Figure(go.Bar(
            x=coefs[idx],
            y=[feat_labels[i] for i in idx],
            orientation='h',
            marker=dict(color=[colors[i] for i in idx], line=dict(width=0)),
            text=[f"{v:.3f}" for v in coefs[idx]],
            textposition='outside',
            textfont=dict(color='rgba(255,255,255,0.55)', size=10),
        ))
        fig_fi.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
            font=dict(color='rgba(255,255,255,0.65)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.06)', color='rgba(255,255,255,0.45)',
                       title="Impact on prediction"),
            yaxis=dict(color='rgba(255,255,255,0.6)'),
            height=320, margin=dict(t=20, b=20, r=60),
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        st.markdown("""
        <div style="font-size:0.78rem; background:rgba(255,255,255,0.03); border-radius:10px; padding:12px 16px; color:rgba(255,255,255,0.4); line-height:1.8;">
        Longer bar = stronger influence on the prediction. Study Hours and Previous Marks tend to have the highest impact.
        </div>
        """, unsafe_allow_html=True)

    # Plain-English explainer
    st.markdown('<div class="sec-title">What do these numbers mean?</div>', unsafe_allow_html=True)
    with st.expander("📖 Plain-English Guide to the Metrics"):
        st.markdown("""
        <div style="color:rgba(255,255,255,0.7); font-size:0.87rem; line-height:2.1;">

        **🎯 Accuracy** — Out of every 100 students, how many did the tool classify correctly?
        An accuracy of 91% means it got 91 right and was wrong on only 9.

        **🔬 Precision** — When the tool says a student will PASS, how often is it actually correct?
        High precision = fewer false alarms (students told they'll pass who actually don't).

        **📡 Recall** — Of all the students who genuinely passed, how many did the tool correctly identify?
        High recall = fewer at-risk students slip through without being flagged.

        **⚖️ F1 Score** — A single number that balances Precision and Recall.
        A high F1 means the tool is reliable in both directions — not just good at one.

        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        📖 Student Result Prediction System &nbsp;·&nbsp; Developed by <b>Dedeepya Thambireddy</b>
    </div>
    """, unsafe_allow_html=True)