import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from pathlib import Path
import base64
from skimage.feature import local_binary_pattern, hog

# Load assets
try:
    flower_path = Path(__file__).parent / "flower.png"
    if flower_path.exists():
        with open(flower_path, "rb") as f:
            flower_base64 = base64.b64encode(f.read()).decode()
    else:
        flower_base64 = ""
except Exception:
    flower_base64 = ""

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

/* Typography setup */
html, body {
    font-family: 'DM Sans', sans-serif;
    color: #2c2c2c;
}

/* Main content: pure white */
.stApp,
.stAppViewContainer,
[data-testid="stAppViewContainer"],
.main,
.block-container,
[data-testid="stAppViewBlockContainer"] {
    background-color: #ffffff !important;
    background: #ffffff !important;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: #1e1e1e;
}

.hero-card {
    background: linear-gradient(135deg, #ffffff 0%, #fce8ee 100%);
    border: 1px solid #f5d0da;
    border-radius: 20px;
    padding: 2.8rem 2rem;
    text-align: center;
    color: #5c3c4b;
    margin-bottom: 2.5rem;
    box-shadow: 0 8px 24px rgba(209, 90, 117, 0.04);
}

.hero-card-title {
    font-family: 'DM Serif Display', serif !important;
    font-size: 3.2rem;
    color: #d15a75 !important;
    margin: 0 0 0.4rem 0;
    line-height: 1.1;
    font-weight: 700;
}

.hero-card-sub {
    font-size: 1.05rem;
    color: #5c3c4b;
    font-weight: 400;
    letter-spacing: 0.03em;
    margin-bottom: 1.4rem;
}

.hero-card-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid #f5d0da;
    border-radius: 30px;
    font-size: 0.78rem;
    color: #d15a75;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.sidebar-tagline {
    font-size: 0.72rem;
    color: #b0697e;
    letter-spacing: 0.08em;
    text-align: center;
    text-transform: uppercase;
    margin-top: -1.2rem;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* Custom Elements */
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

/* Sidebar Specifics */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #fce8ee 100%) !important;
    border-right: 1px solid #f5d0da !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] span {
    color: #3a2030 !important;
}

.sidebar-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #d15a75 !important;
    margin-bottom: 2rem;
    margin-top: 1.5rem;
    text-align: center;
    font-weight: bold;
}

[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    display: none !important;
}

/* Hapus kotak di radio button, biarkan terlihat menyatu */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background-color: transparent !important;
    border: none !important;
    padding: 0.5rem 1rem !important;
    margin-bottom: 0.2rem !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    color: #d15a75 !important;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
    font-weight: 600 !important;
    color: #d15a75 !important;
}

/* Tim Project Section Style */
.section-header {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.8rem;
    color: #1e1e1e;
    border-left: 4px solid #d15a75;
    padding-left: 0.8rem;
    margin-top: 2.5rem;
    margin-bottom: 1.2rem;
    line-height: 1.2;
}

.team-card {
    background-color: #ffffff;
    border: 1px solid #f5d0da;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 12px rgba(209, 90, 117, 0.02);
}

.team-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #d15a75;
    margin-top: 0;
    margin-bottom: 0.8rem;
}

.team-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}

.team-table tr {
    border-bottom: 1px solid #fdf2f4;
}

.team-table tr:last-child {
    border-bottom: none;
}

.team-table td {
    padding: 0.45rem 0.6rem;
    color: #3a2030;
}

.team-table td.name {
    text-align: left;
    font-weight: 500;
    border-right: 1px solid #fdf2f4;
    width: 75%;
    padding-left: 0;
}

.team-table td.nim {
    text-align: right;
    color: #b0697e;
    font-variant-numeric: tabular-nums;
    width: 25%;
    padding-right: 0;
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
    st.markdown("""
    <div class="hero-card">
        <div class="hero-card-title">Skinical</div>
        <div class="hero-card-sub">Skin Lesion Classifier &nbsp;·&nbsp; ISIC 2017 &nbsp;·&nbsp; Classical ML</div>
        <div class="hero-card-badge">Powered by Classical Machine Learning</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Tentang Project
    **Skinical** adalah sistem klasifikasi lesi kulit berbasis web yang dikembangkan menggunakan Machine Learning klasik. Sistem ini bertujuan untuk membantu mendeteksi dini apakah suatu lesi kulit bersifat **Jinak (Benign)** atau **Ganas (Malignant)**.
    
    Aplikasi ini dilatih menggunakan dataset [ISIC 2017 (International Skin Imaging Collaboration)](https://challenge.isic-archive.com/data/#2017) yang merupakan standar benchmark dalam riset analisis citra dermatologis.
    
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

    # Tim Project
    img_html = f"<div style='flex-shrink: 0; width: 85px; height: 85px; display: flex; align-items: center; justify-content: center; margin-left: 0.5rem;'><img src='data:image/png;base64,{flower_base64}' style='width: 100%; height: auto; object-fit: contain;' /></div>" if flower_base64 else ""
    st.markdown(f"""
    <div class="section-header">Tim Project</div>
    <div class="team-card">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="flex: 1; margin-right: 1.2rem;">
                <div class="team-title">Kelompok 5</div>
                <table class="team-table">
                    <tr>
                        <td class="name">Aaron Nikolas Tondosaputro</td>
                        <td class="nim">2802412881</td>
                    </tr>
                    <tr>
                        <td class="name">Albani Kalam Haq</td>
                        <td class="nim">2802498141</td>
                    </tr>
                    <tr>
                        <td class="name">Justin Lysander Setiawan</td>
                        <td class="nim">2802418651</td>
                    </tr>
                    <tr>
                        <td class="name">Kristian Novan</td>
                        <td class="nim">2802458560</td>
                    </tr>
                    <tr>
                        <td class="name">Nadya Salsabila</td>
                        <td class="nim">2802411790</td>
                    </tr>
                    <tr>
                        <td class="name">Sabrina Arfanindia Devi</td>
                        <td class="nim">2802448755</td>
                    </tr>
                </table>
            </div>
            {img_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_demo(model, scaler, bovw_kmeans):
    st.markdown("""
    <div class="hero-card">
        <div class="hero-card-title">Skinical</div>
        <div class="hero-card-sub">Skin Lesion Classifier &nbsp;·&nbsp; ISIC 2017 &nbsp;·&nbsp; Classical ML</div>
        <div class="hero-card-badge">Powered by Classical Machine Learning</div>
    </div>
    """, unsafe_allow_html=True)

    # Upload
    st.markdown("#### Upload Dermoscopic Image")
    uploaded = st.file_uploader(
        "Supported: JPG, PNG, JPEG",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    # Initialize session state for sample image
    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = None

    # Reset selected sample if a new file is uploaded
    if uploaded:
        st.session_state.selected_sample = None

    # Gallery of Sample Images (Always visible below the upload widget)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Or Select a Sample Image")
    col_s1, col_s2, col_s3, _ = st.columns([1, 1, 1, 2.5])
    samples_dir = Path(__file__).parent / "samples"

    with col_s1:
        st.image(str(samples_dir / "sample_1_benign.jpg"), caption="Sample 1", use_container_width=True)
        if st.button("Use Sample 1", key="btn_s1", use_container_width=True):
            st.session_state.selected_sample = str(samples_dir / "sample_1_benign.jpg")
            st.rerun()

    with col_s2:
        st.image(str(samples_dir / "sample_2_benign.jpg"), caption="Sample 2", use_container_width=True)
        if st.button("Use Sample 2", key="btn_s2", use_container_width=True):
            st.session_state.selected_sample = str(samples_dir / "sample_2_benign.jpg")
            st.rerun()

    with col_s3:
        st.image(str(samples_dir / "sample_3_malignant.jpg"), caption="Sample 3", use_container_width=True)
        if st.button("Use Sample 3", key="btn_s3", use_container_width=True):
            st.session_state.selected_sample = str(samples_dir / "sample_3_malignant.jpg")
            st.rerun()

    # Divider before the active image
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Active Image Display & Analysis Section
    has_image = (uploaded is not None) or (st.session_state.selected_sample is not None)

    if has_image:
        if st.session_state.selected_sample:
            filename = Path(st.session_state.selected_sample).name
            friendly_name = "Sample 1" if "sample_1" in filename else "Sample 2" if "sample_2" in filename else "Sample 3"
            st.info(f"Menggunakan gambar sampel: **{friendly_name}**")
            if st.button("Reset Pilihan Gambar"):
                st.session_state.selected_sample = None
                st.rerun()

        # Load image BGR
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            img_bgr    = cv2.imread(st.session_state.selected_sample)

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Image**")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tombol Analyze
        if st.button("Analyze Lesion", type="primary", use_container_width=True):
            with st.spinner("Analyzing and extracting features..."):
                label, prob, img_pre = predict(img_bgr, model, scaler, bovw_kmeans)

            with col2:
                st.markdown("**Feature Extraction Visualizer**")
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Preprocessed",
                    "HOG (Shape)",
                    "LBP (Texture)",
                    "Color Histogram"
                ])
                
                with tab1:
                    st.image(cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Preprocessed Image (CLAHE + Hair Removal)")
                    
                with tab2:
                    from skimage.exposure import rescale_intensity
                    gray_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2GRAY)
                    _, hog_img = hog(gray_pre, orientations=8, pixels_per_cell=(16, 16),
                                     cells_per_block=(2, 2), block_norm='L2-Hys',
                                     visualize=True, feature_vector=True)
                    hog_rescaled = rescale_intensity(hog_img, in_range=(0, 10))
                    st.image(hog_rescaled, use_container_width=True, caption="Histogram of Oriented Gradients (Shape & Edges)")
                    
                with tab3:
                    gray_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2GRAY)
                    lbp_img = local_binary_pattern(gray_pre, P=24, R=3, method='uniform')
                    lbp_norm = np.uint8((lbp_img / lbp_img.max()) * 255) if lbp_img.max() > 0 else np.zeros_like(lbp_img, dtype=np.uint8)
                    st.image(lbp_norm, use_container_width=True, caption="Local Binary Pattern (Texture Micro-patterns)")
                    
                with tab4:
                    r_hist, _ = np.histogram(img_pre[:,:,2], bins=256, range=(0, 256))
                    g_hist, _ = np.histogram(img_pre[:,:,1], bins=256, range=(0, 256))
                    b_hist, _ = np.histogram(img_pre[:,:,0], bins=256, range=(0, 256))
                    
                    import pandas as pd
                    hist_df = pd.DataFrame({
                        'Red Channel': r_hist,
                        'Green Channel': g_hist,
                        'Blue Channel': b_hist
                    })
                    st.line_chart(hist_df, color=["#ff4b4b", "#4beb4b", "#4b4bff"])
                    st.markdown('<p style="text-align: center; color: #8c867e; font-size: 0.8rem; margin-top: 10px;">Intensity Distribution for RGB Channels</p>', unsafe_allow_html=True)

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
            <p style="color:#8c867e; margin:0; font-weight:500;">Upload a dermoscopic image or select a sample image above to begin analysis</p>
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
    st.markdown('<div class="sidebar-tagline">— Skin Lesion Classifier —</div>', unsafe_allow_html=True)
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
<div style="text-align:center; padding: 1rem 0 0.5rem;">
<p style="color:#9e978f; font-size:0.7rem; font-weight:500; letter-spacing:0.05em; margin:0;">
Skinical &nbsp;·&nbsp; COMP7116001 Computer Vision &nbsp;·&nbsp; Kelompok 5 &nbsp;·&nbsp; BINUS University &nbsp;
</p>
<p style="color:#c8b8bc; font-size:0.65rem; margin-top:0.5rem; margin-bottom:0;">© 2026 Kelompok 5 · All rights reserved</p>
</div>
""", unsafe_allow_html=True)