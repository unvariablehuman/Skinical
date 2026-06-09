import streamlit as st
import cv2
import numpy as np
import joblib
import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import base64
import pandas as pd
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

html, body {
    font-family: 'DM Sans', sans-serif;
    color: #2c2c2c;
}

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

/* Hybrid comparison table */
.compare-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    margin-top: 1rem;
}
.compare-table th {
    background: #fce8ee;
    color: #d15a75;
    font-weight: 600;
    padding: 0.6rem 1rem;
    text-align: center;
    border-bottom: 2px solid #f5d0da;
}
.compare-table td {
    padding: 0.55rem 1rem;
    text-align: center;
    border-bottom: 1px solid #f5f0f2;
    color: #2c2c2c;
}
.compare-table tr:last-child td { border-bottom: none; }
.compare-table .best { font-weight: 700; color: #16a34a; }
.compare-table .worst { color: #dc2626; }
.badge-mal {
    background: #ffe3e3; color: #c0392b;
    border-radius: 8px; padding: 2px 10px;
    font-weight: 600; font-size: 0.82rem;
}
.badge-ben {
    background: #dcfce7; color: #15803d;
    border-radius: 8px; padding: 2px 10px;
    font-weight: 600; font-size: 0.82rem;
}
.info-box {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: #0369a1;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_SIZE_CLASSIC = (256, 256)
TARGET_SIZE_HYBRID  = (224, 224)
THRESHOLD_CLASSIC   = 0.30
THRESHOLD_HYBRID    = 0.50
N_CLUSTERS          = 50

# ── Model paths ───────────────────────────────────────────────────────────────
# Classical
MODEL_PATH_CLASSIC  = "Classical ML/rf_skinical.pkl"
SCALER_PATH_CLASSIC = "Classical ML/scaler_skinical.pkl"
BOVW_PATH           = "bovw_kmeans.pkl"

# Hybrid
MODEL_PATH_SVM      = "models/skinical_svm.pkl"
MODEL_PATH_LGBM     = "models/skinical_lgbm.pkl"
MODEL_PATH_RF       = "models/skinical_rf.pkl"
SCALER_PATH_HYBRID  = "models/skinical_scaler.pkl"
PCA_PATH            = "models/skinical_pca.pkl"
EFFNET_PATH         = "models/skinical_effnet.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_classical_models():
    model       = joblib.load(MODEL_PATH_CLASSIC)
    scaler      = joblib.load(SCALER_PATH_CLASSIC)
    bovw_kmeans = joblib.load(BOVW_PATH)
    return model, scaler, bovw_kmeans

@st.cache_resource
def load_hybrid_models():
    svm    = joblib.load(MODEL_PATH_SVM)
    lgbm   = joblib.load(MODEL_PATH_LGBM)
    rf     = joblib.load(MODEL_PATH_RF)
    scaler = joblib.load(SCALER_PATH_HYBRID)
    pca    = joblib.load(PCA_PATH)

    effnet = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
    effnet.load_state_dict(torch.load(EFFNET_PATH, map_location=device))
    effnet = effnet.to(device)
    effnet.eval()

    return {"SVM": svm, "LightGBM": lgbm, "Random Forest": rf}, scaler, pca, effnet

# ── Preprocessing (shared) ────────────────────────────────────────────────────
def remove_hair(img_bgr):
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img_bgr, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

def apply_clahe(img_bgr):
    lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq    = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

def preprocess(img_bgr, size):
    img = remove_hair(img_bgr)
    img = apply_clahe(img)
    return cv2.resize(img, size)

# ── Classical feature extraction ──────────────────────────────────────────────
def extract_lbp(gray, P=24, R=3, n_bins=64):
    lbp  = local_binary_pattern(gray, P=P, R=R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, P+2), density=True)
    return hist

def extract_glcm(gray):
    import mahotas
    return mahotas.features.haralick(gray).mean(axis=0)

def extract_hog_feat(gray):
    return hog(gray, orientations=8, pixels_per_cell=(16,16),
               cells_per_block=(2,2), block_norm="L2-Hys", feature_vector=True)

def extract_lab_hist(img_bgr, bins=32):
    lab   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    feats = []
    for ch, (lo, hi) in enumerate([(0,100),(-128,127),(-128,127)]):
        h, _ = np.histogram(lab[:,:,ch].ravel(), bins=bins, range=(lo, hi), density=True)
        feats.append(h)
    return np.concatenate(feats)

def extract_hsv_hist(img_bgr, bins=32):
    hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    feats = []
    for ch, (lo, hi) in enumerate([(0,180),(0,255),(0,255)]):
        h, _ = np.histogram(hsv[:,:,ch].ravel(), bins=bins, range=(lo, hi), density=True)
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

def extract_classical_features(img_bgr, bovw_kmeans):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return np.concatenate([
        extract_lbp(gray),
        extract_glcm(gray),
        extract_hog_feat(gray),
        extract_lab_hist(img_bgr),
        extract_hsv_hist(img_bgr),
        extract_bovw(img_bgr, bovw_kmeans)
    ])

# ── Hybrid feature extraction ─────────────────────────────────────────────────
dl_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract_deep_features(img_bgr, effnet):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor  = dl_transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = effnet(tensor)
    return feat.squeeze().cpu().numpy()

def extract_hybrid_feat_only(img_bgr, effnet):
    """Classical features for hybrid (LBP + HOG + LAB) without BoVW/GLCM."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lbp  = extract_lbp(gray)
    hog_f = hog(gray, orientations=9, pixels_per_cell=(16,16),
                cells_per_block=(2,2), block_norm="L2-Hys", feature_vector=True)
    lab  = extract_lab_hist(img_bgr)
    deep = extract_deep_features(img_bgr, effnet)
    return np.concatenate([deep, lbp, hog_f, lab])

# ── Predict functions ─────────────────────────────────────────────────────────
def predict_classical(img_bgr, model, scaler, bovw_kmeans):
    img     = preprocess(img_bgr, TARGET_SIZE_CLASSIC)
    feat    = extract_classical_features(img, bovw_kmeans).reshape(1, -1)
    feat    = np.nan_to_num(feat)
    feat_sc = scaler.transform(feat)
    prob    = model.predict_proba(feat_sc)[0, 1]
    label   = "Malignant" if prob >= THRESHOLD_CLASSIC else "Benign"
    return label, prob, img

def predict_hybrid_all(img_bgr, classifiers, scaler, pca, effnet):
    """Run all 3 hybrid classifiers and return results dict."""
    img  = preprocess(img_bgr, TARGET_SIZE_HYBRID)
    feat = extract_hybrid_feat_only(img, effnet).reshape(1, -1)
    feat = np.nan_to_num(feat)
    feat_sc = scaler.transform(feat)
    feat_pca = pca.transform(feat_sc)

    results = {}
    for name, clf in classifiers.items():
        prob  = clf.predict_proba(feat_pca)[0, 1]
        label = "Malignant" if prob >= THRESHOLD_HYBRID else "Benign"
        results[name] = {"label": label, "prob": prob}
    return results, img

# ── Shared image visualizer ───────────────────────────────────────────────────
def show_feature_visualizer(img_pre):
    tab1, tab2, tab3, tab4 = st.tabs([
        "Preprocessed", "HOG (Shape)", "LBP (Texture)", "Color Histogram"
    ])
    with tab1:
        st.image(cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB),
                 use_container_width=True,
                 caption="Preprocessed Image (CLAHE + Hair Removal)")
    with tab2:
        from skimage.exposure import rescale_intensity
        gray_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2GRAY)
        _, hog_img = hog(gray_pre, orientations=8, pixels_per_cell=(16,16),
                         cells_per_block=(2,2), block_norm="L2-Hys",
                         visualize=True, feature_vector=True)
        hog_rescaled = rescale_intensity(hog_img, in_range=(0, 10))
        st.image(hog_rescaled, use_container_width=True,
                 caption="Histogram of Oriented Gradients")
    with tab3:
        gray_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2GRAY)
        lbp_img  = local_binary_pattern(gray_pre, P=24, R=3, method="uniform")
        lbp_norm = np.uint8((lbp_img / lbp_img.max()) * 255) if lbp_img.max() > 0 else np.zeros_like(lbp_img, dtype=np.uint8)
        st.image(lbp_norm, use_container_width=True,
                 caption="Local Binary Pattern (Texture)")
    with tab4:
        r_hist, _ = np.histogram(img_pre[:,:,2], bins=256, range=(0,256))
        g_hist, _ = np.histogram(img_pre[:,:,1], bins=256, range=(0,256))
        b_hist, _ = np.histogram(img_pre[:,:,0], bins=256, range=(0,256))
        hist_df = pd.DataFrame({
            "Red Channel": r_hist,
            "Green Channel": g_hist,
            "Blue Channel": b_hist
        })
        st.line_chart(hist_df, color=["#ff4b4b","#4beb4b","#4b4bff"])

# ── Pages ─────────────────────────────────────────────────────────────────────
def show_description():
    st.markdown("""
    <div class="hero-card">
        <div class="hero-card-title">Skinical</div>
        <div class="hero-card-sub">Skin Lesion Classifier &nbsp;·&nbsp; Classical ML &amp; Hybrid DL</div>
        <div class="hero-card-badge">Classical ML + EfficientNetB3 Hybrid</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Tentang Project
    **Skinical** adalah sistem klasifikasi lesi kulit berbasis web yang dikembangkan menggunakan dua pendekatan: **Machine Learning Klasik** dan **Hybrid Deep Learning + Classical ML**. Sistem ini bertujuan untuk membantu mendeteksi dini apakah suatu lesi kulit bersifat **Jinak (Benign)** atau **Ganas (Malignant)**.

    ---

    ### Dua Mode Klasifikasi

    **🔵 Classical ML** — Dilatih menggunakan dataset [ISIC 2017](https://challenge.isic-archive.com/data/#2017). Pipeline menggunakan ekstraksi fitur tradisional (LBP, GLCM, HOG, LAB/HSV Histogram, BoVW) dengan Random Forest sebagai classifier.

    **🟣 Hybrid DL + Classical ML** — Menggabungkan deep features dari EfficientNetB3 (1280-dim) dengan classical features (LBP + HOG + LAB) lalu diklasifikasikan menggunakan SVM, LightGBM, dan Random Forest.

    ---

    ### Alur Kerja Hybrid
    ```
    Image → Hair Removal + CLAHE
          ├── EfficientNetB3 (pretrained) → 1280-dim deep features
          └── LBP + HOG + LAB            → classical features
                   ↓ Concatenate + StandardScaler + PCA
          SVM / LightGBM / Random Forest → Benign / Malignant
    ```

    ---

    ### Performa Model
    | Model | Accuracy | AUC | F1 |
    |---|---|---|---|
    | Classical ML (RF) | ~67% | 0.736 | 0.470 |
    | Hybrid SVM | **84.4%** | **0.923** | **0.828** |
    | Hybrid LightGBM | 83.8% | 0.920 | 0.818 |
    | Hybrid Random Forest | 80.5% | 0.903 | 0.775 |
    """)

    st.markdown("""
    <div class="warning-box">
        ⚠️ <strong>Penting:</strong> Aplikasi ini dirancang sebagai alat bantu edukasi dan penelitian awal. Hasil klasifikasi model tidak boleh dijadikan satu-satunya rujukan diagnosis medis. Selalu konsultasikan dengan dokter spesialis kulit (dermatolog) berlisensi.
    </div>
    """, unsafe_allow_html=True)

    img_html = f"<div style='flex-shrink:0;width:85px;height:85px;display:flex;align-items:center;justify-content:center;margin-left:0.5rem;'><img src='data:image/png;base64,{flower_base64}' style='width:100%;height:auto;object-fit:contain;'/></div>" if flower_base64 else ""
    st.markdown(f"""
    <div class="section-header">Tim Project</div>
    <div class="team-card">
        <div style="display:flex;align-items:center;justify-content:space-between;">
            <div style="flex:1;margin-right:1.2rem;">
                <div class="team-title">Kelompok 5</div>
                <table class="team-table">
                    <tr><td class="name">Aaron Nikolas Tondosaputro</td><td class="nim">2802412881</td></tr>
                    <tr><td class="name">Albani Kalam Haq</td><td class="nim">2802498141</td></tr>
                    <tr><td class="name">Justin Lysander Setiawan</td><td class="nim">2802418651</td></tr>
                    <tr><td class="name">Kristian Novan</td><td class="nim">2802458560</td></tr>
                    <tr><td class="name">Nadya Salsabila</td><td class="nim">2802411790</td></tr>
                    <tr><td class="name">Sabrina Arfanindia Devi</td><td class="nim">2802448755</td></tr>
                </table>
            </div>
            {img_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_demo_classical(model, scaler, bovw_kmeans):
    st.markdown("""
    <div class="hero-card">
        <div class="hero-card-title">Skinical</div>
        <div class="hero-card-sub">Classical ML &nbsp;·&nbsp; ISIC 2017 &nbsp;·&nbsp; Random Forest</div>
        <div class="hero-card-badge">Powered by Classical Machine Learning</div>
    </div>
    """, unsafe_allow_html=True)

    _run_demo(
        mode="classical",
        predict_fn=lambda img: predict_classical(img, model, scaler, bovw_kmeans),
        threshold=THRESHOLD_CLASSIC
    )


def show_demo_hybrid(classifiers, scaler, pca, effnet):
    st.markdown("""
    <div class="hero-card">
        <div class="hero-card-title">Skinical</div>
        <div class="hero-card-sub">Hybrid DL + Classical ML &nbsp;·&nbsp; EfficientNetB3</div>
        <div class="hero-card-badge">Powered by EfficientNetB3 + Classical Features</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        ℹ️ Mode <strong>Hybrid</strong> menjalankan 3 classifier sekaligus (SVM, LightGBM, Random Forest) dan menampilkan perbandingan hasilnya.
    </div>
    """, unsafe_allow_html=True)

    _run_demo(
        mode="hybrid",
        predict_fn=lambda img: predict_hybrid_all(img, classifiers, scaler, pca, effnet),
        threshold=THRESHOLD_HYBRID
    )


def _run_demo(mode, predict_fn, threshold):
    st.markdown("#### Upload Dermoscopic Image")
    uploaded = st.file_uploader(
        "Supported: JPG, PNG, JPEG",
        type=["jpg","jpeg","png"],
        label_visibility="collapsed",
        key=f"uploader_{mode}"
    )

    if f"selected_sample_{mode}" not in st.session_state:
        st.session_state[f"selected_sample_{mode}"] = None
    if uploaded:
        st.session_state[f"selected_sample_{mode}"] = None

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Or Select a Sample Image")
    col_s1, col_s2, col_s3, _ = st.columns([1,1,1,2.5])
    samples_dir = Path(__file__).parent / "samples"

    with col_s1:
        st.image(str(samples_dir / "sample_1_benign.jpg"), caption="Sample 1", use_container_width=True)
        if st.button("Use Sample 1", key=f"btn_s1_{mode}", use_container_width=True):
            st.session_state[f"selected_sample_{mode}"] = str(samples_dir / "sample_1_benign.jpg")
            st.rerun()
    with col_s2:
        st.image(str(samples_dir / "sample_2_benign.jpg"), caption="Sample 2", use_container_width=True)
        if st.button("Use Sample 2", key=f"btn_s2_{mode}", use_container_width=True):
            st.session_state[f"selected_sample_{mode}"] = str(samples_dir / "sample_2_benign.jpg")
            st.rerun()
    with col_s3:
        st.image(str(samples_dir / "sample_3_malignant.jpg"), caption="Sample 3", use_container_width=True)
        if st.button("Use Sample 3", key=f"btn_s3_{mode}", use_container_width=True):
            st.session_state[f"selected_sample_{mode}"] = str(samples_dir / "sample_3_malignant.jpg")
            st.rerun()

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    has_image = uploaded is not None or st.session_state[f"selected_sample_{mode}"] is not None

    if has_image:
        if st.session_state[f"selected_sample_{mode}"]:
            filename = Path(st.session_state[f"selected_sample_{mode}"]).name
            friendly = "Sample 1" if "sample_1" in filename else "Sample 2" if "sample_2" in filename else "Sample 3"
            st.info(f"Menggunakan gambar sampel: **{friendly}**")
            if st.button("Reset Pilihan Gambar", key=f"reset_{mode}"):
                st.session_state[f"selected_sample_{mode}"] = None
                st.rerun()

        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            img_bgr = cv2.imread(st.session_state[f"selected_sample_{mode}"])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Image**")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Analyze Lesion", type="primary", use_container_width=True, key=f"analyze_{mode}"):
            with st.spinner("Analyzing and extracting features..."):
                result = predict_fn(img_bgr)

            # ── Classical result ───────────────────────────────────────────────
            if mode == "classical":
                label, prob, img_pre = result
                with col2:
                    st.markdown("**Feature Extraction Visualizer**")
                    show_feature_visualizer(img_pre)

                st.markdown('<hr class="divider">', unsafe_allow_html=True)
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

                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Probability</div><div class="metric-value">{prob:.1%}</div></div>', unsafe_allow_html=True)
                with c2:
                    conf = abs(prob - 0.5) * 2
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Confidence</div><div class="metric-value">{conf:.1%}</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Threshold</div><div class="metric-value">{threshold}</div></div>', unsafe_allow_html=True)

            # ── Hybrid result ──────────────────────────────────────────────────
            else:
                results_dict, img_pre = result
                with col2:
                    st.markdown("**Feature Extraction Visualizer**")
                    show_feature_visualizer(img_pre)

                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown("#### 🔬 Hasil Komparasi 3 Classifier")

                # Majority vote
                votes     = [v["label"] for v in results_dict.values()]
                final_lbl = max(set(votes), key=votes.count)
                avg_prob  = np.mean([v["prob"] for v in results_dict.values()])
                is_mal    = final_lbl == "Malignant"
                css_cls   = "result-malignant" if is_mal else "result-benign"
                emoji     = "⚠️" if is_mal else "✅"
                color     = "#f87171" if is_mal else "#4ade80"

                st.markdown(f"""
                <div class="{css_cls}">
                    <div class="result-label" style="color:{color}">{emoji} {final_lbl}</div>
                    <div class="result-prob">Majority vote dari 3 model &nbsp;·&nbsp; Rata-rata prob: {avg_prob:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Comparison table
                best_prob = max(results_dict.values(), key=lambda x: x["prob"])["prob"]
                rows = ""
                for clf_name, res in results_dict.items():
                    badge_cls = "badge-mal" if res["label"] == "Malignant" else "badge-ben"
                    prob_cls  = "best" if res["prob"] == best_prob else ""
                    rows += f"""
                    <tr>
                        <td><strong>{clf_name}</strong></td>
                        <td><span class="{badge_cls}">{res["label"]}</span></td>
                        <td class="{prob_cls}">{res["prob"]:.4f}</td>
                        <td class="{prob_cls}">{res["prob"]:.1%}</td>
                    </tr>"""

                st.markdown(f"""
                <table class="compare-table">
                    <thead>
                        <tr>
                            <th>Classifier</th>
                            <th>Prediksi</th>
                            <th>Raw Prob</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Probability</div><div class="metric-value">{avg_prob:.1%}</div></div>', unsafe_allow_html=True)
                with c2:
                    conf = abs(avg_prob - 0.5) * 2
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Confidence</div><div class="metric-value">{conf:.1%}</div></div>', unsafe_allow_html=True)
                with c3:
                    agree = "✓ Semua Setuju" if len(set(votes)) == 1 else f"⚡ {votes.count('Malignant')}–{votes.count('Benign')} Split"
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Konsensus</div><div class="metric-value" style="font-size:1.1rem;">{agree}</div></div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="warning-box">
                ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only
                and is not a substitute for professional medical diagnosis.
                Always consult a qualified dermatologist.
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="border:1px dashed #d1cbbd;border-radius:12px;padding:2.5rem;text-align:center;background-color:#ffffff;">
            <p style="color:#8c867e;margin:0;font-weight:500;">Upload a dermoscopic image or select a sample image above to begin analysis</p>
        </div>
        """, unsafe_allow_html=True)


# ── Load models ───────────────────────────────────────────────────────────────
try:
    hybrid_classifiers, hybrid_scaler, hybrid_pca, hybrid_effnet = load_hybrid_models()
    hybrid_ok = True
except Exception as e:
    hybrid_ok  = False
    hybrid_err = str(e)

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Skinical</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">— Skin Lesion Classifier —</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigasi",
        ["Deskripsi", "Demo Hybrid DL"],
        label_visibility="collapsed"
    )

if page == "Deskripsi":
    show_description()
elif page == "Demo Hybrid DL":
    if hybrid_ok:
        show_demo_hybrid(hybrid_classifiers, hybrid_scaler, hybrid_pca, hybrid_effnet)
    else:
        st.error(f"Gagal load Hybrid model: {hybrid_err}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:1rem 0 0.5rem;">
<p style="color:#9e978f;font-size:0.7rem;font-weight:500;letter-spacing:0.05em;margin:0;">
Skinical &nbsp;·&nbsp; COMP7116001 Computer Vision &nbsp;·&nbsp; Kelompok 5 &nbsp;·&nbsp; BINUS University
</p>
<p style="color:#c8b8bc;font-size:0.65rem;margin-top:0.5rem;margin-bottom:0;">© 2026 Kelompok 5 · All rights reserved</p>
</div>
""", unsafe_allow_html=True)