# 🔬 Skinical

**Hybrid Skin Lesion Classification using EfficientNetB3 and Handcrafted Feature Fusion**
COMP7116001 — Computer Vision · BINUS University · 2026

**Final Project — Group 5**

**Supervisor:**
Dr. Fiqri Ramadhan Tambunan

---

## Overview

Skinical is a hybrid framework for binary skin lesion classification using dermoscopic images from the **ISIC 2017 Challenge dataset**. This project was developed as the final project for the Computer Vision course at BINUS University.

The proposed framework combines deep feature representations extracted from **EfficientNetB3** with handcrafted visual descriptors, including **Local Binary Patterns (LBP), Histogram of Oriented Gradients (HOG), and LAB color histograms**. The integration of semantic deep features and explicit visual descriptors aims to improve the discrimination capability between benign and malignant skin lesions.

---

## Team Members — Group 5

| Name                       | Contribution                                                        |
| -------------------------- | ------------------------------------------------------------------- |
| Aaron Nikolas Tondosaputro | Results and Analysis                                                |
| Nadya Salsabila            | Introduction and Related Work                                       |
| Sabrina Arfanindia Devi    | Methodology, Hybrid Pipeline, Feature Extraction, Model Development |
| Albani Kalam Haq           | Discussion, Limitations, and Streamlit Deployment                   |
| Kristian Novan             | Video Demonstration                                                 |
| Justin Lysander Setiawan   | Conclusion and Future Work                                          |

**Supervisor:**
Dr. Fiqri Ramadhan Tambunan

---

## Pipeline

```
Dermoscopic Image
        │
        ▼
Preprocessing
 ├── Hair Removal (DullRazor)
 ├── CLAHE Enhancement (LAB Color Space)
 └── Resize → 224×224
        │
        ▼
Dual Feature Extraction
 ├── EfficientNetB3 Deep Features
 │       └── 1,536 dimensions
 │
 └── Handcrafted Features
         ├── LBP
         ├── HOG
         └── LAB Histogram
             └── 6,244 dimensions
        │
        ▼
Feature Fusion
        │
        ▼
StandardScaler Normalization
        │
        ▼
PCA Dimensionality Reduction
        │
        ▼
Classifier
 ├── SVM
 ├── LightGBM
 └── Random Forest
        │
        ▼
Prediction
Benign / Malignant
```

---

## Results

Evaluation was conducted on the **ISIC 2017 Challenge test set**.

| Classifier    | Accuracy  | F1-score  | ROC-AUC   |
| ------------- | --------- | --------- | --------- |
| Random Forest | 80.5%     | 0.775     | 0.903     |
| LightGBM      | 83.8%     | 0.818     | 0.920     |
| **SVM**       | **84.4%** | **0.828** | **0.923** |

The hybrid SVM classifier achieved the best performance with the highest ROC-AUC value of **0.923**, demonstrating the effectiveness of combining deep representations with handcrafted visual descriptors.

---

## Dataset

**ISIC 2017 Challenge Dataset**

The dataset contains dermoscopic images categorized into two classes:

* Benign lesions
* Malignant lesions

Dataset source:

https://challenge.isic-archive.com/data/#2017

---


## Run Locally

```bash
git clone https://github.com/unvariablehuman/Skinical.git
cd Skinical
pip install -r requirements.txt
streamlit run app.py
```

---

## Resources

* GitHub Repository:
  https://github.com/unvariablehuman/Skinical

* Presentation Materials:
  https://canva.link/ccytf8m7ivy64n8

* Demo Video:
  https://drive.google.com/drive/folders/1lk6efINcuAxPOEoOyXHWBwztVKySC9os?usp=sharing

* Supplementary Materials:
  https://drive.google.com/drive/folders/1pyvyDu4OkUTSgECoMG8wVlH_vy-yXcB2

---

## Limitations

* The framework is evaluated only for binary classification between benign and malignant lesions.
* Validation on larger and independent clinical datasets is required to assess generalization capability.
* The hybrid feature extraction pipeline introduces additional computational cost compared with single-stream approaches.

---

*COMP7116001 Computer Vision · Group 5 · BINUS University · 2026*
