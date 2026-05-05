# Kini nga import nag-enable sa mas modern nga type hints behavior sa Python
from __future__ import annotations

# Built-in libraries
import io  # para sa in-memory file handling (e.g. BytesIO)
import os  # para sa OS operations (paths, directories)
import sys  # para ma-manipulate ang Python path
from pathlib import Path  # mas clean nga file path handling

# External libraries
import matplotlib.pyplot as plt  # para sa plotting/graphs
import pandas as pd  # para sa data manipulation (DataFrame)
import streamlit as st  # para sa UI sa web app

# Kuhaon ang ROOT directory (parent folder sa project)
ROOT = Path(__file__).resolve().parents[1]

# Kung wala pa ang ROOT sa Python path, i-add siya
# Para maka-import ta ug modules gikan sa src folder
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Import gikan sa imong custom modules (src folder)

# Function para kuhaon ang top keywords gikan sa text data
from src.keyword_utils import extract_top_keywords

# Function para automatic detect kung asa ang text column sa dataset
from src.nlp_utils import detect_text_column

# Mga function para sa sentiment prediction pipeline
from src.predictor import (
    confidence_band,       # muhatag ug label (High/Medium/Low) based sa confidence score
    predict_dataframe,     # main function para mag-predict sa sentiment sa dataframe
    train_or_load_pipeline # mag-train or load existing ML model
)
st.set_page_config(page_title="AI Sentiment Studio", page_icon="📊", layout="wide")

CUSTOM_CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.stApp {
    background: linear-gradient(180deg, #0b1020 0%, #111827 45%, #0f172a 100%);
    color: #e5e7eb;
}
.block-container {
    max-width: 1280px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}
section[data-testid="stSidebar"] {display: none !important;}
header[data-testid="stHeader"] {background: transparent;}
div[data-testid="stToolbar"] {right: 1rem;}
.hero-card, .panel {
    background: rgba(17, 24, 39, 0.82);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 20px;
    padding: 1.25rem 1.25rem 1rem 1.25rem;
    box-shadow: 0 12px 40px rgba(0,0,0,0.28);
}
.metric-card {
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 18px;
    padding: 1rem 1.1rem;
}
.metric-label {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-bottom: 0.25rem;
}
.metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f8fafc;
}
.badge {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-left: 0.35rem;
}
.badge-high {background: rgba(34,197,94,0.18); color: #86efac; border: 1px solid rgba(34,197,94,0.35);}
.badge-medium {background: rgba(245,158,11,0.18); color: #fcd34d; border: 1px solid rgba(245,158,11,0.35);}
.badge-low {background: rgba(239,68,68,0.18); color: #fca5a5; border: 1px solid rgba(239,68,68,0.35);}
.section-title {
    font-size: 1.08rem;
    font-weight: 700;
    margin-bottom: 0.9rem;
    color: #f8fafc;
}
.small-note {
    color: #94a3b8;
    font-size: 0.92rem;
}
.upload-wrap {
    border: 1px dashed rgba(148, 163, 184, 0.35);
    border-radius: 18px;
    padding: 0.7rem;
    background: rgba(15, 23, 42, 0.45);
}
hr {
    border: none;
    border-top: 1px solid rgba(148, 163, 184, 0.15);
    margin: 1rem 0 0.5rem 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_resource
# Function para i-load ang ML pipeline (model)
def load_pipeline():
    # Tawagon ang train_or_load_pipeline
    # force_retrain=False → kung naa nay existing model, i-load ra (dili na mag-train balik)
    return train_or_load_pipeline(force_retrain=False)


# Function para basahon ang uploaded file gikan sa Streamlit
def read_uploaded_file(uploaded_file):
    # Kuhaon ang file extension (.csv, .xlsx, etc.)
    suffix = Path(uploaded_file.name).suffix.lower()
    
    # Kung Excel file (.xlsx or .xls)
    if suffix in [".xlsx", ".xls"]:
        # gamiton ang pandas read_excel
        return pd.read_excel(uploaded_file)
    
    # Default: kung dili Excel, assume CSV
    return pd.read_csv(uploaded_file)


# Function para himuon ang downloadable Excel file gikan sa DataFrame
def make_download_bytes(df: pd.DataFrame):
    # Create ug in-memory buffer (wala mag-save sa disk)
    buffer = io.BytesIO()
    
    # Gamit ExcelWriter para ma-convert ang DataFrame to Excel format
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        # Isulat ang DataFrame ngadto sa Excel sheet
        df.to_excel(writer, index=False, sheet_name="predictions")
    
    # I-reset ang pointer sa buffer ngadto sa start (importante para ma-read)
    buffer.seek(0)
    
    # I-return ang bytes → gamiton ni sa download button sa Streamlit
    return buffer.getvalue()


# Function para mag-generate HTML badge based sa confidence score
def badge_html(score: float) -> str:
    # Kuhaon ang confidence level (High / Medium / Low)
    band = confidence_band(score)
    
    # Mapping sa class name para sa CSS styling
    # e.g. High → badge-high
    klass = {"High": "high", "Medium": "medium", "Low": "low"}[band]
    
    # Return HTML string nga naay CSS class
    # gamiton ni sa UI para colorful badge
    return f'<span class="badge badge-{klass}">{band} confidence</span>'

# Header
st.markdown(
    """
    <div class="hero-card">
        <div style="display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; flex-wrap:wrap;">
            <div>
                <div style="font-size:2rem; font-weight:800; color:#f8fafc;">AI Sentiment Studio</div>
                <div class="small-note" style="margin-top:0.35rem;">
                    Professional sentiment analysis using <b>machine learning</b> and <b>NLP</b>.
                    Upload text-only Excel or CSV files and get positive, negative, or neutral predictions.
                </div>
            </div>
            <div style="text-align:right; min-width:200px;">
                <div class="small-note">Prediction engine</div>
                <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">Local ML + NLP</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
col_a, col_b = st.columns([1.15, 0.85], gap="large")

with col_a:
    st.markdown('<div class="panel"><div class="section-title">Upload Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-wrap">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Excel or CSV", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="small-note">The app auto-detects the main text column. No sentiment labels are required.</div></div>',
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        """
        <div class="panel">
            <div class="section-title">Workflow</div>
            <div class="small-note">
                1. Upload a text-only file.<br>
                2. The app cleans the text with NLP preprocessing.<br>
                3. A local machine learning model predicts sentiment.<br>
                4. Review charts and download the final report.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if uploaded is not None:
    df = read_uploaded_file(uploaded)
    if df.empty:
        st.error("The uploaded file is empty.")
        st.stop()

    text_col = detect_text_column(df)
    if text_col is None:
        st.error("No text-like column was found in the uploaded file.")
        st.stop()

    with st.spinner("Analyzing text with NLP + machine learning..."):
        pipeline = load_pipeline()
        results = predict_dataframe(df, text_col)

    total_rows = len(results)
    pos_count = int((results["predicted_sentiment"] == "positive").sum())
    neg_count = int((results["predicted_sentiment"] == "negative").sum())
    neu_count = int((results["predicted_sentiment"] == "neutral").sum())
    avg_conf = float(results["confidence"].mean()) if total_rows else 0.0

    m1, m2, m3, m4 = st.columns(4, gap="medium")
    metrics = [
        ("Rows analyzed", f"{total_rows:,}"),
        ("Positive", f"{pos_count:,}"),
        ("Negative", f"{neg_count:,}"),
        ("Average confidence", f"{avg_conf:.0%}"),
    ]
    for col, (label, value) in zip([m1,m2,m3,m4], metrics):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True
            )

    st.write("")
    c1, c2 = st.columns([1.05, 0.95], gap="large")

    with c1:
        st.markdown('<div class="panel"><div class="section-title">Sentiment Distribution</div>', unsafe_allow_html=True)
        counts = results["predicted_sentiment"].value_counts().reindex(["positive", "negative", "neutral"]).fillna(0)
        fig, ax = plt.subplots(figsize=(7.2, 4.1))
        bars = ax.bar(counts.index, counts.values)
        ax.set_ylabel("Count")
        ax.set_xlabel("Sentiment")
        ax.set_title("Predicted Sentiment Breakdown")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_facecolor("white")
        st.pyplot(fig, clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel"><div class="section-title">Report Summary</div>', unsafe_allow_html=True)
        dominant = counts.idxmax() if counts.sum() > 0 else "neutral"
        neutral_share = f"{(neu_count / total_rows):.0%}" if total_rows else "0%"
        st.markdown(
            f"""
            <div class="small-note">
                <b>Detected text column:</b> {text_col}<br><br>
                <b>Dominant sentiment:</b> {dominant.title()}<br>
                <b>Neutral share:</b> {neutral_share}<br>
                <b>Rows with high confidence:</b> {(results["confidence"] >= 0.75).sum():,}<br>
                <b>Rows with medium confidence:</b> {((results["confidence"] >= 0.55) & (results["confidence"] < 0.75)).sum():,}<br>
                <b>Rows with low confidence:</b> {(results["confidence"] < 0.55).sum():,}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="small-note">Confidence scale is based on the model probability after rule-based sentiment adjustment.</div></div>', unsafe_allow_html=True)

    st.write("")
    k1, k2 = st.columns(2, gap="large")
    with k1:
        st.markdown('<div class="panel"><div class="section-title">Top Positive Keywords</div>', unsafe_allow_html=True)
        pos_terms = extract_top_keywords(results, text_col, "predicted_sentiment", "positive", top_n=10)
        if pos_terms:
            pos_df = pd.DataFrame(pos_terms, columns=["keyword", "count"]).sort_values("count", ascending=True)
            fig, ax = plt.subplots(figsize=(7, 4.6))
            ax.barh(pos_df["keyword"], pos_df["count"])
            ax.set_xlabel("Count")
            ax.set_title("Most Frequent Positive Terms")
            ax.grid(axis="x", linestyle="--", alpha=0.3)
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("No positive keywords found in this file.")
        st.markdown("</div>", unsafe_allow_html=True)

    with k2:
        st.markdown('<div class="panel"><div class="section-title">Top Negative Keywords</div>', unsafe_allow_html=True)
        neg_terms = extract_top_keywords(results, text_col, "predicted_sentiment", "negative", top_n=10)
        if neg_terms:
            neg_df = pd.DataFrame(neg_terms, columns=["keyword", "count"]).sort_values("count", ascending=True)
            fig, ax = plt.subplots(figsize=(7, 4.6))
            ax.barh(neg_df["keyword"], neg_df["count"])
            ax.set_xlabel("Count")
            ax.set_title("Most Frequent Negative Terms")
            ax.grid(axis="x", linestyle="--", alpha=0.3)
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("No negative keywords found in this file.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="panel"><div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
    preview = results.copy()
    preview.insert(preview.columns.get_loc("confidence")+1, "confidence_band", preview["confidence"].apply(confidence_band))
    show_cols = [text_col, "predicted_sentiment", "confidence", "confidence_band"]
    extra_cols = [c for c in preview.columns if c not in show_cols][:0]
    st.dataframe(preview[show_cols + extra_cols], use_container_width=True, height=420)
    st.markdown("</div>", unsafe_allow_html=True)

    download_bytes = make_download_bytes(preview)
    st.download_button(
        "Download prediction report (.xlsx)",
        data=download_bytes,
        file_name="sentiment_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
else:
    st.write("")
    st.info("Upload an Excel or CSV file to start the sentiment analysis.")
