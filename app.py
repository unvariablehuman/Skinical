import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from pathlib import Path
from skimage.feature import local_binary_pattern, hog

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skinical",
    page_icon="🔬",
    layout="centered"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #faf9f6;
    color: #1e1e1e;
}

.stApp { background-color: #faf9f6; }

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: #1e1e1e;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    color: #1e1e1e;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-size: 1rem;
    color: #666666;
    font-weight: 300;
    letter-spacing: 0.05em;
    margin-bottom: 2rem;
}

.metric-card {
    background: #f0eee9;
    border: 1px solid #e2ded5;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}

.metric-label {
    font-size: 0.75rem;
    color: #77726a;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #1e1e1e;
}

.result-malignant {
    background: linear-gradient(135deg, #fff5f5, #ffe3e3);
    border: 1px solid #fecaca;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}

.result-benign {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 1px solid #bbf7d0;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}

.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.result-prob {
    font-size: 0.9rem;
    color: #555555;
    letter-spacing: 0.05em;
}

.warning-box {
    background: #fffbeb;
    border: 1px solid #fef3c7;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: #b45309;
    margin-top: 1.5rem;
}

.info-chip {
    display: inline-block;
    background: #f0eee9;
    border: 1px solid #e2ded5;
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-size: 0.78rem;
    color: #555555;
    margin: 0.2rem;
}

.divider {
    border: none;
    border-top: 1px solid #e2ded5;
    margin: 2rem 0;
}

.stButton > button {
    background: #1e1e1e !important;
    color: #faf9f6 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.6rem 2rem !important;
    width: 100%;
}

.stButton > button:hover {
    background: #333333 !important;
}

.upload-area {
    border: 1px dashed #d1cbbd;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_SIZE  = (256, 256)
THRESHOLD    = 0.30
N_CLUSTERS   = 50
MODEL_PATH   = "rf_skinical.pkl"
SCALER_PATH  = "scaler_skinical.pkl"
BOVW_PATH    = "bovw_kmeans.pkl"

# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model       = joblib.load(MODEL_PATH)
    scaler      = joblib.load(SCALER_PATH)
    bovw_kmeans = joblib.load(BOVW_PATH)
    return model, scaler, bovw_kmeans

# ── Preprocessing ─────────────────────────────────────────────────────────────
def remove_hair(img_bgr):
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    cleaned  = cv2.inpaint(img_bgr, hair_mask, inpaintRadius=3,
                           flags=cv2.INPAINT_TELEA)
    return cleaned

def apply_clahe(img_bgr):
    lab      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b  = cv2.split(lab)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq     = clahe.apply(l)
    enhanced = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def preprocess(img_bgr, size=TARGET_SIZE):
    img = remove_hair(img_bgr)
    img = apply_clahe(img)
    img = cv2.resize(img, size)
    return img

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_lbp(gray, P=24, R=3, n_bins=64):
    lbp  = local_binary_pattern(gray, P=P, R=R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                            range=(0, P+2), density=True)
    return hist

def extract_glcm(gray):
    import mahotas
    return mahotas.features.haralick(gray).mean(axis=0)

def extract_hog_feat(gray):
    return hog(gray, orientations=8, pixels_per_cell=(16,16),
               cells_per_block=(2,2), block_norm='L2-Hys',
               feature_vector=True)

def extract_lab_hist(img_bgr, bins=32):
    lab   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    feats = []
    for ch, (lo, hi) in enumerate([(0,100),(-128,127),(-128,127)]):
        h, _ = np.histogram(lab[:,:,ch].ravel(), bins=bins,
                             range=(lo, hi), density=True)
        feats.append(h)
    return np.concatenate(feats)

def extract_hsv_hist(img_bgr, bins=32):
    hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
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
    gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lbp_feat  = extract_lbp(gray)
    glcm_feat = extract_glcm(gray)
    hog_feat  = extract_hog_feat(gray)
    lab_feat  = extract_lab_hist(img_bgr)
    hsv_feat  = extract_hsv_hist(img_bgr)
    bovw_feat = extract_bovw(img_bgr, bovw_kmeans)
    return np.concatenate([lbp_feat, glcm_feat, hog_feat,
                           lab_feat, hsv_feat, bovw_feat])

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(img_bgr, model, scaler, bovw_kmeans, threshold=THRESHOLD):
    img     = preprocess(img_bgr)
    feat    = extract_features(img, bovw_kmeans).reshape(1, -1)
    feat    = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    feat_sc = scaler.transform(feat)
    prob    = model.predict_proba(feat_sc)[0, 1]
    label   = "Malignant" if prob >= threshold else "Benign"
    return label, prob, img

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Skinical</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">SKIN LESION CLASSIFIER · ISIC 2017 · CLASSICAL ML</div>',
            unsafe_allow_html=True)



st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Load models
try:
    model, scaler, bovw_kmeans = load_models()
    st.markdown('<p style="color:#4a7c59; font-size:0.8rem;">● Model loaded</p>',
                unsafe_allow_html=True)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Upload
st.markdown("#### Upload Dermoscopic Image")
uploaded = st.file_uploader(
    "Supported: JPG, PNG, JPEG",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded:
    # Convert to BGR
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original**")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                 use_container_width=True)

    with st.spinner("Analyzing..."):
        label, prob, img_pre = predict(img_bgr, model, scaler, bovw_kmeans)

    with col2:
        st.markdown("**Preprocessed**")
        st.image(cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB),
                 use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Result
    is_mal  = label == "Malignant"
    css_cls = "result-malignant" if is_mal else "result-benign"
    emoji   = "⚠️" if is_mal else "✅"
    color   = "#f87171" if is_mal else "#4ade80"

    st.markdown(f"""
    <div class="{css_cls}">
        <div class="result-label" style="color:{color}">{emoji} {label}</div>
        <div class="result-prob">Malignant probability: {prob:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Probability</div>
            <div class="metric-value">{prob:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        conf = abs(prob - 0.5) * 2
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{conf:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Threshold</div>
            <div class="metric-value">{THRESHOLD}</div>
        </div>""", unsafe_allow_html=True)

    # Warning
    st.markdown("""
    <div class="warning-box">
        ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only
        and is not a substitute for professional medical diagnosis.
        Always consult a qualified dermatologist.
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="upload-area">
        <p style="color:#444; margin:0">Upload a dermoscopic image to begin analysis</p>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<p style="color:#777; font-size:0.75rem; text-align:center;">
Skinical · COMP7116001 Computer Vision · BINUS University · ISIC 2017
</p>
""", unsafe_allow_html=True)
