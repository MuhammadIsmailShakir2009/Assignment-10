"""
NeuralForge — Visual CNN Trainer
Streamlit multi-page application
"""

import streamlit as st

st.set_page_config(
    page_title="NeuralForge — CNN Trainer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111118;
    border-right: 1px solid #2a2a3a;
}
[data-testid="stSidebar"] * { color: #e8e8f0 !important; }

/* Main background */
.stApp { background: #0a0a0f; }
.main .block-container { padding-top: 1.5rem; }

/* Metric boxes */
[data-testid="stMetric"] {
    background: #1a1a24;
    border: 1px solid #2a2a3a;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    color: #06b6d4 !important;
}
[data-testid="stMetricLabel"] { color: #6b6b80 !important; }

/* Buttons */
.stButton > button {
    background: #7c3aed;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 0.55rem 1.4rem;
    transition: all 0.2s;
}
.stButton > button:hover { background: #6d28d9; transform: translateY(-1px); }

/* Headers */
h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; color: #e8e8f0 !important; }
h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; color: #e8e8f0 !important; }

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #6b6b80;
}
[data-testid="stTabs"] button[aria-selected="true"] { color: #06b6d4; border-bottom-color: #06b6d4; }

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed #2a2a3a;
    border-radius: 10px;
    background: #111118;
}

/* Progress bar */
.stProgress > div > div { background-color: #7c3aed; }

/* Expander */
[data-testid="stExpander"] { background: #111118; border: 1px solid #2a2a3a; border-radius: 8px; }

/* DataFrame */
[data-testid="stDataFrame"] { border: 1px solid #2a2a3a; border-radius: 8px; }

/* Info / success / warning boxes */
.stAlert { border-radius: 8px; font-family: 'Space Mono', monospace; font-size: 0.78rem; }

/* Selectbox, slider */
[data-testid="stSelectbox"] select, .stSlider { background: #111118; }

.mono { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: #6b6b80; }
.accent { color: #7c3aed; }
.accent2 { color: #06b6d4; }
.success { color: #10b981; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 NeuralForge")
    st.markdown('<p class="mono">// browser-native CNN trainer</p>', unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation",
        ["📁  Dataset Builder", "🏋️  Train Model", "📊  Evaluate", "🔮  Predict"],
        label_visibility="collapsed",
    )

    st.divider()

    # Session state summary
    n_classes = len(st.session_state.get("classes", {}))
    total_imgs = sum(
        len(v) for v in st.session_state.get("classes", {}).values()
    )
    model_trained = st.session_state.get("model_trained", False)

    st.markdown("**Session Status**")
    col1, col2 = st.columns(2)
    col1.metric("Classes", n_classes)
    col2.metric("Images", total_imgs)
    st.metric("Model", "✅ Ready" if model_trained else "⬜ Not trained")

    st.divider()
    st.markdown('<p class="mono">TensorFlow · Keras · scikit-learn<br>NumPy · Matplotlib · Seaborn</p>', unsafe_allow_html=True)

# ── Route Pages ──────────────────────────────────────────────────────────────
if "📁" in page:
    from pages import dataset
    dataset.render()
elif "🏋️" in page:
    from pages import train
    train.render()
elif "📊" in page:
    from pages import evaluate
    evaluate.render()
elif "🔮" in page:
    from pages import predict
    predict.render()
