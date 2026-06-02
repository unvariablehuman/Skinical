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
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html {
    background: linear-gradient(135deg, #fff5f6 0%, #f3f0ff 100%) !important;
}

body {
    font-family: 'DM Sans', sans-serif;
    background: transparent !important;
    color: #2c2c2c;
}

[class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp,
.stAppViewContainer,
.stAppViewContainer > section,
.stAppViewContainer > section > div,
.main,
.main > div,
.block-container {
    background: transparent !important;
}


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
    color: #736d65;
    font-weight: 400;
    letter-spacing: 0.05em;
    margin-bottom: 2rem;
}

.metric-card {
    background: #ffffff;
    border: 1px solid #e2ded5;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
}

.metric-label {
    font-size: 0.75rem;
    color: #8c867e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
    font-weight: 500;
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
    font-size: 0.95rem;
    color: #555555;
    letter-spacing: 0.02em;
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

.divider {
    border: none;
    border-top: 1px solid #e2ded5;
    margin: 2rem 0;
}

/* Sidebar styling - Modern Minimalist */
[data-testid="stSidebar"] {
    /* Gradasi putih ke pink yang sangat lembut di bawah */
    background: linear-gradient(180deg, #ffffff 0%, #fff0f2 100%) !important;
    border-right: 1px solid #f2e1e3 !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] li {
    color: #333333 !important;
}

.sidebar-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #d15a75 !important; /* Aksen dark pink elegan */
    margin-bottom: 2rem;
    margin-top: 1.5rem;
    text-align: center;
    font-weight: bold;
}

[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    display: none !important;
}

[data-testid="stSidebar"] .stRadio > div {
    background-color: transparent !important;
}

/* Hapus bentuk card dari navigasi radio button */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background-color: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1rem !important;
    margin-bottom: 0.4rem !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background-color: #fff6f7 !important;
    color: #d15a75 !important;
}

/* Style untuk menu yang sedang dipilih (aktif) */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
    background-color: transparent !important;
    font-weight: 600 !important;
    color: #d15a75 !important;
}

/* Ubah warna bullet radio button saat dipilih jadi pink */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) div[data-baseweb="radio"] div {
    background-color: #d15a75 !important;
    border-color: #d15a75 !important;
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

# ── Pages ─────────────────────────────────────────────────────────────────────
def show_description():
    st.markdown('<div class="hero-title">Skinical</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">SKIN LESION CLASSIFIER · ISIC 2017 · CLASSICAL ML</div>',
                unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    st.markdown("""
    ### Tentang Project
    **Skinical** adalah sistem klasifikasi lesi kulit berbasis web yang dikembangkan menggunakan Machine Learning klasik. Sistem ini bertujuan untuk membantu mendeteksi dini apakah suatu lesi kulit bersifat **Jinak (Benign)** atau **Ganas (Malignant)**.
    
    Aplikasi ini dilatih menggunakan dataset **ISIC 2017** (International Skin Imaging Collaboration) yang merupakan standar benchmark dalam riset analisis citra dermatologis.
    
    ---
    
    ### Alur Kerja & Ekstraksi Fitur
    Sebelum melakukan klasifikasi, citra dermoskopik melalui proses preprocessing dan ekstraksi fitur yang komprehensif:
    
    1. **Preprocessing Citra**:
       - *Hair Removal*: Menghilangkan rambut pada kulit yang menghalangi lesi menggunakan metode morfologi Blackhat dan Inpainting.
       - *Contrast Enhancement*: Menggunakan CLAHE (Contrast Limited Adaptive Histogram Equalization) pada ruang warna LAB untuk memperjelas batas lesi.
    
    2. **Ekstraksi Fitur Multi-dimensi**:
       - **Tekstur (LBP)**: *Local Binary Pattern* digunakan untuk mengekstrak pola mikro-tekstur permukaan lesi.
       - **Tekstur Spasial (GLCM)**: Menggunakan fitur Haralick untuk menangkap hubungan spasial intensitas piksel.
       - **Bentuk (HOG)**: *Histogram of Oriented Gradients* mengekstrak fitur bentuk dan kontur tepi lesi.
       - **Warna (LAB & HSV)**: Histogram warna pada ruang warna LAB dan HSV untuk menangkap gradasi warna lesi.
       - **Fitur Lokal (BoVW)**: *Bag of Visual Words* dengan algoritma ORB untuk merepresentasikan pola visual penting pada lesi.
       
    ---
    
    ### Informasi & Performa Model
    Berikut adalah detail model klasifikasi yang digunakan di balik layar:
    
    - **Model**: Random Forest
    - **AUC ROC**: 0.736
    - **Recall**: 0.680
    - **F1 Malignant**: 0.470
    - **Threshold**: 0.30
    """)
    
    st.markdown("""
    <br>
    <div class="warning-box">
        ⚠️ <strong>Penting:</strong> Aplikasi ini dirancang sebagai alat bantu edukasi dan penelitian awal. Hasil klasifikasi model tidak boleh dijadikan satu-satunya rujukan diagnosis medis. Selalu konsultasikan dengan dokter spesialis kulit (dermatolog) berlisensi.
    </div>
    """, unsafe_allow_html=True)

def show_demo(model, scaler, bovw_kmeans):
    st.markdown('<div class="hero-title">Skinical</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">SKIN LESION CLASSIFIER · ISIC 2017 · CLASSICAL ML</div>',
                unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<p style="color:#d15a75; font-size:0.8rem; font-weight:600;">● Model loaded successfully</p>',
                unsafe_allow_html=True)

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
        <div style="border: 1px dashed #d1cbbd; border-radius: 12px; padding: 2.5rem; text-align: center; background-color: #ffffff;">
            <p style="color:#8c867e; margin:0; font-weight:500;">Upload a dermoscopic image to begin analysis</p>
        </div>
        """, unsafe_allow_html=True)

# ── UI ────────────────────────────────────────────────────────────────────────
# Load models
try:
    model, scaler, bovw_kmeans = load_models()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Sidebar Navigation
with st.sidebar:
    st.markdown('<div class="sidebar-title">Skinical</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigasi",
        ["Deskripsi", "Demo Model"],
        label_visibility="collapsed"
    )

if page == "Deskripsi":
    show_description()
elif page == "Demo Model":
    show_demo(model, scaler, bovw_kmeans)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<p style="color:#9e978f; font-size:0.75rem; text-align:center; font-weight:500;">
Skinical · COMP7116001 Computer Vision · BINUS University · ISIC 2017
</p>
""", unsafe_allow_html=True)