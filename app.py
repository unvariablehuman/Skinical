"""
Skinical — Skin Lesion Classification
Streamlit prediction page using BoVW + RF pipeline.

Required files (same directory as this script):
    bovw_kmeans.pkl
    rf_skinical.pkl
    scaler_skinical.pkl

Run:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
import joblib
from pathlib import Path
from PIL import Image
import io

# ── Optional deps ──────────────────────────────────────────────────────────────
try:
    import mahotas
    MAHOTAS_OK = True
except ImportError:
    MAHOTAS_OK = False

from skimage.feature import local_binary_pattern, hog
from skimage.feature import graycomatrix, graycoprops

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skinical · Skin Lesion Classifier",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

:root {
    --bg: #F5F2ED;
    --card: #FFFDF9;
    --border: #E0D9CF;
    --ink: #1C1917;
    --muted: #78716C;
    --accent: #C2410C;       /* warning/malignant */
    --safe: #15803D;         /* benign */
    --highlight: #F97316;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--ink);
}

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
    letter-spacing: -0.5px;
}

/* hero */
.hero {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
}
.hero .badge {
    display: inline-block;
    background: var(--ink);
    color: var(--bg);
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 2px;
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: 3.2rem;
    margin: 0 0 0.5rem;
    line-height: 1.05;
}
.hero .subtitle {
    color: var(--muted);
    font-size: 1rem;
    font-weight: 300;
    max-width: 420px;
    margin: 0 auto 2rem;
    line-height: 1.6;
}

/* upload area styling */
.stFileUploader > div > div {
    border: 2px dashed var(--border) !important;
    border-radius: 8px !important;
    background: var(--card) !important;
    transition: border-color 0.2s;
}
.stFileUploader > div > div:hover {
    border-color: var(--ink) !important;
}

/* result cards */
.result-card {
    border-radius: 10px;
    padding: 1.6rem 2rem;
    margin-top: 1.5rem;
    border: 1.5px solid var(--border);
    background: var(--card);
}
.result-card.malignant {
    border-color: var(--accent);
    background: #FFF7F5;
}
.result-card.benign {
    border-color: var(--safe);
    background: #F0FDF4;
}
.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.result-label.malignant { color: var(--accent); }
.result-label.benign    { color: var(--safe);   }
.result-sub {
    font-size: 0.85rem;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    margin-bottom: 1rem;
}

/* probability bar */
.prob-bar-outer {
    height: 8px;
    background: var(--border);
    border-radius: 99px;
    overflow: hidden;
    margin: 0.5rem 0 0.3rem;
}
.prob-bar-inner {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
}
.prob-bar-inner.malignant { background: var(--accent); }
.prob-bar-inner.benign    { background: var(--safe);   }

.prob-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
}

/* disclaimer */
.disclaimer {
    font-size: 0.78rem;
    color: var(--muted);
    background: #FEF3C7;
    border: 1px solid #FDE68A;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin-top: 2rem;
    line-height: 1.5;
}

/* pipeline steps */
.pipeline {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 1rem 0;
    justify-content: center;
}
.pipe-step {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    background: var(--ink);
    color: var(--bg);
    padding: 3px 10px;
    border-radius: 3px;
    letter-spacing: 0.5px;
}
.pipe-arrow { color: var(--muted); font-size: 0.9rem; line-height: 2; }

.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR = Path(__file__).parent

@st.cache_resource(show_spinner="Loading models…")
def load_models():
    rf      = joblib.load(MODEL_DIR / "rf_skinical.pkl")
    scaler  = joblib.load(MODEL_DIR / "scaler_skinical.pkl")
    bovw_km = joblib.load(MODEL_DIR / "bovw_kmeans.pkl")
    return rf, scaler, bovw_km

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing  (mirrors notebook)
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SIZE = (256, 256)

def remove_hair(img_bgr):
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    bh     = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(bh, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

def apply_clahe(img_bgr):
    lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    merged = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def preprocess(img_bgr):
    img = remove_hair(img_bgr)
    img = apply_clahe(img)
    return cv2.resize(img, TARGET_SIZE)

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction  (mirrors notebook)
# ─────────────────────────────────────────────────────────────────────────────

N_CLUSTERS = 50

def extract_lbp(gray, P=24, R=3, n_bins=64):
    lbp  = local_binary_pattern(gray, P=P, R=R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, P+2), density=True)
    return hist

def extract_glcm(gray):
    if MAHOTAS_OK:
        return mahotas.features.haralick(gray).mean(axis=0)
    # Fallback: skimage GLCM → 13 placeholder features
    gcm   = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                          levels=256, symmetric=True, normed=True)
    props = ['contrast','dissimilarity','homogeneity','energy',
             'correlation','ASM']
    feats = np.concatenate([[graycoprops(gcm, p).mean()] for p in props])
    # Pad / trim to 13 to match training dimensions
    out = np.zeros(13)
    out[:len(feats)] = feats[:13]
    return out

def extract_hog_feat(gray):
    return hog(gray, orientations=8, pixels_per_cell=(16,16),
               cells_per_block=(2,2), block_norm='L2-Hys',
               feature_vector=True)

def extract_lab_hist(img_bgr, bins=32):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    feats = []
    for ch, (lo, hi) in enumerate([(0,100),(-128,127),(-128,127)]):
        h, _ = np.histogram(lab[:,:,ch].ravel(), bins=bins,
                             range=(lo, hi), density=True)
        feats.append(h)
    return np.concatenate(feats)

def extract_hsv_hist(img_bgr, bins=32):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    feats = []
    for ch, (lo, hi) in enumerate([(0,180),(0,255),(0,255)]):
        h, _ = np.histogram(hsv[:,:,ch].ravel(), bins=bins,
                             range=(lo, hi), density=True)
        feats.append(h)
    return np.concatenate(feats)

def extract_bovw(img_bgr, kmeans, n_clusters=N_CLUSTERS):
    orb  = cv2.ORB_create(nfeatures=500)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, des = orb.detectAndCompute(gray, None)
    hist = np.zeros(n_clusters)
    if des is not None:
        labels = kmeans.predict(des)
        for l in labels:
            hist[l] += 1
        hist = hist / (hist.sum() + 1e-7)
    return hist

def extract_features(img_bgr, bovw_kmeans):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return np.concatenate([
        extract_lbp(gray),
        extract_glcm(gray),
        extract_hog_feat(gray),
        extract_lab_hist(img_bgr),
        extract_hsv_hist(img_bgr),
        extract_bovw(img_bgr, bovw_kmeans),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict(pil_img, rf, scaler, bovw_km, threshold=0.5):
    # PIL → BGR numpy
    img_rgb  = np.array(pil_img.convert("RGB"))
    img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    # Preprocess
    img_pre  = preprocess(img_bgr)
    # Feature extraction
    feat     = extract_features(img_pre, bovw_km).reshape(1, -1)
    feat     = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    feat_sc  = scaler.transform(feat)
    # Predict
    prob_mal = float(rf.predict_proba(feat_sc)[0, 1])
    label    = "Malignant" if prob_mal >= threshold else "Benign"
    return label, prob_mal, img_pre

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

# Hero header
st.markdown("""
<div class="hero">
    <div class="badge">ISIC 2017 · Binary Classification</div>
    <h1>Skinical</h1>
    <p class="subtitle">Upload a dermoscopy image to classify it as
    <em>benign</em> or <em>malignant</em> using a Random Forest trained on
    hand-crafted skin-lesion features.</p>
    <div class="pipeline">
        <span class="pipe-step">Hair Removal</span>
        <span class="pipe-arrow">→</span>
        <span class="pipe-step">CLAHE</span>
        <span class="pipe-arrow">→</span>
        <span class="pipe-step">LBP</span>
        <span class="pipe-arrow">+</span>
        <span class="pipe-step">GLCM</span>
        <span class="pipe-arrow">+</span>
        <span class="pipe-step">HOG</span>
        <span class="pipe-arrow">+</span>
        <span class="pipe-step">LAB Hist</span>
        <span class="pipe-arrow">+</span>
        <span class="pipe-step">BoVW</span>
        <span class="pipe-arrow">→</span>
        <span class="pipe-step">Random Forest</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Load models
try:
    rf, scaler, bovw_km = load_models()
    models_ok = True
except Exception as e:
    st.error(f"**Could not load model files.**  \n{e}  \n\nMake sure `bovw_kmeans.pkl`, `rf_skinical.pkl`, and `scaler_skinical.pkl` are in the same directory as `app.py`.")
    models_ok = False

# Settings expander
with st.expander("⚙️  Settings", expanded=False):
    threshold = st.slider(
        "Classification threshold (malignant probability ≥ threshold → Malignant)",
        min_value=0.10, max_value=0.90, value=0.50, step=0.01,
        format="%.2f"
    )
    show_preprocessed = st.checkbox("Show preprocessed image", value=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# Upload
uploaded = st.file_uploader(
    "Drop a dermoscopy image here  ·  JPG / PNG / JPEG",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)

if uploaded and models_ok:
    pil_img = Image.open(uploaded)

    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.markdown("**Original**")
        st.image(pil_img, use_container_width=True)

    with st.spinner("Extracting features and predicting…"):
        label, prob_mal, img_pre = predict(pil_img, rf, scaler, bovw_km, threshold)

    prob_ben = 1.0 - prob_mal
    css_cls  = label.lower()          # "malignant" | "benign"
    bar_pct  = int(prob_mal * 100)

    if show_preprocessed:
        with col2:
            st.markdown("**Preprocessed (hair removal + CLAHE)**")
            import cv2
            pre_rgb = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)
            st.image(pre_rgb, use_container_width=True)

    # Result card
    icon = "⚠️" if label == "Malignant" else "✅"
    st.markdown(f"""
    <div class="result-card {css_cls}">
        <div class="result-label {css_cls}">{icon} {label}</div>
        <div class="result-sub">malignant probability · threshold = {threshold:.2f}</div>
        <div class="prob-bar-outer">
            <div class="prob-bar-inner {css_cls}" style="width:{bar_pct}%"></div>
        </div>
        <div class="prob-label">
            P(malignant) = <strong>{prob_mal:.4f}</strong>
            &nbsp;·&nbsp;
            P(benign) = <strong>{prob_ben:.4f}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Detailed probability breakdown
    st.markdown("")
    st.markdown("#### Probability breakdown")
    pcol1, pcol2 = st.columns(2)
    pcol1.metric("P(Malignant)", f"{prob_mal:.4f}", delta=None)
    pcol2.metric("P(Benign)",    f"{prob_ben:.4f}", delta=None)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
    ⚕️  <strong>Medical disclaimer:</strong> This tool is a research prototype trained on the ISIC 2017 dataset.
    It is <em>not</em> a certified medical device and must <em>not</em> be used for clinical diagnosis.
    Always consult a qualified dermatologist for any skin concern.
    </div>
    """, unsafe_allow_html=True)

elif not uploaded:
    st.info("Upload a dermoscopy image above to get a prediction.")
