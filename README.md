# Forecasting Missing-Children Caseloads and Stratifying State Tracing Regimes in India via Classical Machine Learning

**Author:** Aayush Sapra (Roll No.: `24570096`)  
**Course:** B.Sc. (Hons.) Computer Science, 3rd Year (Semester VI), University of Delhi  
**Subject:** Machine Learning | **Supervisor:** Nikhil Sir  
**Problem ID:** Problem No. 20 (*Child Protection & Public Safety Analytics* — **UN SDG 16** Target 16.2 & **UN SDG 5** Target 5.2)

---

## 📌 Project Overview

Over **83,000 children** are reported missing annually in India, with girls accounting for more than **75%** of newly registered First Information Reports (FIRs). Because state child-protection agencies allocate recovery resources retrospectively, this repository implements a **leakage-free classical machine-learning pipeline** on official National Crime Records Bureau (NCRB) *Crime in India* panels (2016–2022) across **36 States and Union Territories** (`6,048` long-format records; `1,008` State–Gender–Year cohorts; `424` consecutive annual transitions).

This study addresses two core machine-learning tasks using strictly syllabus-approved algorithms:
1. **Supervised One-Year-Ahead Caseload Regression ($\text{Missing}_{t+1}$):** Benchmarks **Simple Linear Regression (SLR)**, **Multiple Linear Regression (MLR)**, **Ridge Regression ($L_2$)**, **Lasso Regression ($L_1$)**, **Degree-2 Polynomial Regression + Ridge**, and a **2-Layer Artificial Neural Network (ANN, 64–32 ReLU)** against a naive persistence baseline under 4-fold forward-chaining (`TimeSeriesSplit`) cross-validation.
2. **Unsupervised State Stratification ($k \in \{2,\dots,6\}$):** Compares **$k$-Means Clustering (`k-means++`)** and **Ward's Hierarchical Agglomerative Clustering** on four standardised multi-year indicators (Male Tracing Rate, Female Tracing Rate, Female Share of Newly Missing Children, and Untraced Backlog Share).

---

## 📂 Repository Structure

```text
├── ncrb_missing_children.csv                      # Compiled NCRB / Dataful dataset (6,048 rows, 36 States/UTs)
├── run_experiments.py                             # Main ML pipeline, CV tuning & 7 publication figures generator
├── per_state.py                                   # Per-State/UT test error & algorithmic equity audit (Table A1)
├── Missing_Children_ML_Assignment_Aayush_Sapra.pdf # Final 5-page two-column journal paper (PDF)
├── requirements.txt                               # Python dependencies
├── README.md                                      # Project overview, benchmark tables & run instructions
└── figures/                                       # Generated publication-grade PDF figures (Fig. 1 to Fig. 7)
```

---

## 📊 Summary of Experimental Results

### 1. One-Year-Ahead Missing-Children Regression (`2020–2022` Test Set, $n = 144$)

| Regression Model | Test MAE | Test RMSE | Test $R^2$ | 4-Fold CV-RMSE | Key Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Persistence Baseline ($\hat{y}_{t+1} = \text{Missing}_t$)** | 213.08 | 449.32 | 0.942 | — | Naive no-change benchmark |
| **Simple Linear Regression (SLR)** | 216.60 | 496.32 | 0.930 | **307.81** | Lowest CV error (1 predictor), but underfits post-2020 surges |
| **Multiple Linear Regression (MLR)** | 272.06 | 468.68 | 0.937 | 639.40 | High variance due to exact collinearity among NCRB counts |
| **Ridge Regression ($\alpha = 0.1$)** | 244.24 | 436.16 | 0.946 | 352.11 | Stabilises collinear weights; beats persistence |
| **Lasso Regression ($\alpha = 10$)** | 231.52 | 484.48 | 0.933 | 315.87 | Zeroes out 5 redundant collinear features |
| **Polynomial Regression (Deg 2 + Ridge, $\alpha = 10$)** | 204.00 | **361.06** | **0.963** | 437.35 | **Best RMSE (19.6% below persistence)** via quadratic interactions |
| **Artificial Neural Network (64–32 ReLU, $\alpha = 1.0$)** | **201.23** | 394.01 | 0.956 | 380.69 | **Best MAE**; highly stable under 95% PCA compression |

---

### 2. Unsupervised State Stratification ($k = 3$ Operational Policy Tiers, $N = 35$ Active States/UTs)

* **$k = 2$ Global Optimum:** $k$-Means Silhouette = **0.425** ($\text{CH} = 34.84, \text{DB} = 0.880$) vs. Ward Hierarchical Silhouette = **0.403**.
* **$k = 3$ Policy Tiers:** $k$-Means Silhouette = **0.358** ($\text{CH} = 27.76, \text{DB} = 0.909$):

| Operational Tier ($k = 3$) | States ($n$) | Male Tracing % | Female Tracing % | Female Share % | Backlog Share % | Representative States / UTs |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Tier 1: High-Recovery / Low-Backlog** | 18 | **75.4%** | **81.6%** | 66.8% | **21.3%** | Kerala, Telangana, Tamil Nadu, Rajasthan, Madhya Pradesh, Andhra Pradesh, Chhattisgarh |
| **Tier 2: High-Backlog / Female-Skewed Priority** | 7 | **38.3%** | **44.2%** | **79.6%** | **61.6%** | **West Bengal, Bihar, Odisha, Punjab, Jammu & Kashmir, Chandigarh, Ladakh** |
| **Tier 3: Moderate-Recovery / Chronic-Backlog** | 10 | 52.7% | 57.7% | 58.6% | 47.5% | Delhi, Maharashtra, Uttar Pradesh, Haryana, Karnataka, Jharkhand, Uttarakhand |

---

### 3. PCA Ablation & 2020 COVID-19 Lockdown Stress Test

| Experimental Setting | MAE | RMSE | $R^2$ |
| :--- | :---: | :---: | :---: |
| Multiple Linear Regression, All Features (`2020–2022`) | 272.06 | 468.68 | 0.937 |
| Multiple Linear Regression, PCA 95% (8 PCs) | 350.96 | 656.25 | 0.877 |
| Ridge ($\alpha = 0.1$), All Features | 244.24 | 436.16 | 0.946 |
| Ridge ($\alpha = 0.1$), PCA 95% (8 PCs) | 326.32 | 627.75 | 0.888 |
| ANN (64–32, $\alpha = 1$), All Features | 201.23 | 394.01 | 0.956 |
| **ANN (64–32, $\alpha = 1$), PCA 95% (8 PCs)** | **201.08** | 411.09 | 0.952 |
| **2020 Lockdown (`2019 -> 2020`): Persistence Baseline** | **228.09** | **413.63** | **0.912** |
| 2020 Lockdown (`2019 -> 2020`): Ridge ($\alpha = 0.1$) | 276.82 | 455.28 | 0.893 |
| 2020 Lockdown (`2019 -> 2020`): Poly-2 + Ridge | 293.19 | 526.59 | 0.857 |

---

## 🚀 How to Run the Code Locally

1. **Clone the repository and install dependencies:**
   ```bash
   git clone https://github.com/aayushsapra200611/missing-children-ml-india.git
   cd missing-children-ml-india
   pip install -r requirements.txt
   ```

2. **Run the main experiment pipeline & regenerate all 7 PDF figures:**
   ```bash
   python run_experiments.py
   ```

3. **Run the per-State/UT test error and algorithmic equity audit (Appendix Table A1):**
   ```bash
   python per_state.py
   ```

---

## 📚 Data Sources & Citation

* **Primary Data:** National Crime Records Bureau (NCRB), *Crime in India (2016–2022), Chapter 15: Missing and Traced Persons*, Ministry of Home Affairs, Government of India. Compiled by Dataful ([Dataset ID 18468](https://dataful.in/datasets/18468/)) and verified against [Lok Sabha Unstarred Question No. 4003 Annexures](https://www.mha.gov.in/MHA1/Par2017/pdfs/par2025-pdfs/LS25032025/4003.pdf).
