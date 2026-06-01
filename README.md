# 🔬 Skinical

**Skin Lesion Binary Classification using Classical Machine Learning**  
COMP7116001 — Computer Vision · BINUS University · 2026

---

## Overview

Skinical is a classical ML pipeline for binary skin lesion classification (malignant vs benign) built on the **ISIC 2017** dermoscopic image dataset. The project is part of the Computer Vision final project at BINUS University, with a strict constraint of **no deep learning**.

The pipeline integrates hair removal, contrast enhancement, multi-feature handcrafted extraction, class imbalance handling, and threshold-tuned evaluation — following best practices from recent dermatological ML literature.

---

## Pipeline

```
Raw Image
    │
    ▼
Hair Removal (DullRazor-lite)
    │
    ▼
CLAHE Enhancement
    │
    ▼
Resize → 256×256
    │
    ▼
Feature Extraction
  ├── LBP   (Local Binary Patterns)       → 64 features
  ├── GLCM  (Haralick Texture)            → 13 features
  ├── HOG   (Histogram of Gradients)      → variable
  ├── LAB   (Color Histogram)             → 96 features
  ├── HSV   (Color Histogram)             → 96 features
  └── BoVW  (ORB + Bag of Visual Words)   → 50 features
    │
    ▼
Z-score Normalization (StandardScaler)
    │
    ▼
Random Oversampling (minority class)
    │
    ▼
Random Forest Classifier
    │
    ▼
Threshold Tuning (0.30)
    │
    ▼
Prediction: Benign / Malignant
```

---

## Results

Evaluated on ISIC 2017 test set (600 images, 117 malignant / 483 benign):

| Metric | Value |
|---|---|
| **ROC-AUC** | **0.736** |
| Recall Malignant | 0.68 |
| F1 Malignant | 0.47 |
| MCC | 0.315 |
| Accuracy | 0.71 |

Two operating thresholds are reported:

| Mode | Threshold | Recall Malignant | F1 Malignant | Use Case |
|---|---|---|---|---|
| Balanced | 0.30 | 0.68 | 0.47 | General screening |
| High Sensitivity | 0.15 | ~0.92 | ~0.35 | Clinical safety |

> AUC-ROC is the primary metric. Accuracy is not used as the primary metric due to 4.3:1 class imbalance.

---

## Dataset

**ISIC 2017 Skin Lesion Analysis Toward Melanoma Detection**

| Split | Total | Malignant | Benign |
|---|---|---|---|
| Train | 2,000 | 374 (18.7%) | 1,626 (81.3%) |
| Validation | 150 | 30 | 120 |
| Test | 600 | 117 | 483 |

Labels: `melanoma == 1` → malignant, otherwise benign.

Dataset available on [ISIC CHALLENGE 2017](https://challenge.isic-archive.com/data/#2017).

---

## References

- Pattnaik et al. (2025). *Skin Lesion Image Classification With Tree-Based Ensembles: Benchmarking Random Forest and Gradient Boosting.* Cureus 17(9): e92432.
- Thomas, E.K. (2021). *Prediction of Malignant Melanoma using Machine Learning.* MSc Thesis, National College of Ireland.
- ISIC 2017 Challenge: https://challenge.isic-archive.com/landing/2017/


---

## Run Locally

```bash
git clone https://github.com/unvariablehuman/Skinical.git
cd Skinical
pip install -r requirements.txt
streamlit run app.py
```

---

## Limitations

- Classical ML inherently limited for dermoscopic classification — deep learning (CNN) expected to yield significantly higher AUC (0.85–0.94)
- Pipeline does not implement ABCD rule features (asymmetry, border, color variation, diameter)
- Validation set small (n=30 malignant), test set results are primary performance reference
- Model trained and evaluated on ISIC 2017 only — generalizability to other populations not validated

---

*COMP7116001 Computer Vision · Group Project · BINUS University · 2026*
