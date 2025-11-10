# 🧠 ML Assignments – PPGI • UFPB | 2025

A modular repository for machine learning exercises and experiments.
Developed as part of the **Machine Learning** discipline (MSc in Informatics – PPGI/UFPB).

---

## 📂 Project Structure
```
ML/
├── configs/              # YAML configuration files
├── notebooks/            # Jupyter notebooks for each task
├── outputs/              # Generated results and plots
├── src/                  # Core implementation
│   ├── ml/
│   │   ├── data.py       # Dataset generation utilities
│   │   ├── viz.py        # Visualization tools
│   │   └── metrics.py    # Evaluation and metrics functions
│   └── tasks/            # Specific ML algorithms (KNN, SVM, MLP, etc.)
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 🧩 Features
- Synthetic dataset generation — bivariate Gaussian distributions
- Config-driven workflows via YAML
- Modular architecture for reproducible ML tasks

---

## ⚙️ Setup
```bash
uv sync
```

---

## 📊 Results Overview

> **Data Configuration (GLOBAL)**
>
> - Train/Test split: `test_size = 0.20`, `random_state = 42`, `stratify = y`

### 🌀 Unsupervised Models
| Model | Notebook | Primary Metric(s) | Value | Notes |
|:--|:--|:--|:--|:--|
| **K-means** | [10-kmeans.ipynb](./notebooks/10-kmeans.ipynb) | Adjusted Rand Index (ARI) | **0.9342** | Confusion matrix shown in notebook; decision regions plotted. |
| **FCM (Fuzzy C-Means)** | [11-fcm.ipynb](./notebooks/11-fcm.ipynb) | Centers / Iterations | Centers ≈ `[[60.07, 30.89], [24.51, 9.80]]`; Iter = **15** | Membership heatmap and clusters visualization. |

### ✅ Supervised Models
| Model | Notebook | Accuracy | Precision | Recall | F1 |
|:--|:--|--:|--:|--:|--:|
| **Logistic Regression** | [20-logreg.ipynb](./notebooks/20-logreg.ipynb) | **0.9833** | **0.9677** | **1.0000** | **0.9836** |
| **ANN (MLPClassifier)** | [21-ann.ipynb](./notebooks/21-ann.ipynb) | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **SVM (RBF)** | [22-svm.ipynb](./notebooks/22-svm.ipynb) | **0.9833** | **0.9677** | **1.0000** | **0.9836** |
| **Decision Tree** | [23-tree.ipynb](./notebooks/23-tree.ipynb) | **0.9500** | **0.9655** | **0.9333** | **0.9492** |
| **K-NN (best k = 3)** | [24-knn.ipynb](./notebooks/24-knn.ipynb) | **0.9833** | **0.9677** | **1.0000** | **0.9836** |

> *Notes:* Metrics reported above refer to the **test split** unless otherwise stated. Each notebook includes the decision boundary plot and the corresponding confusion matrix.

---

## 👨‍💻 Author
**Lucas G. Dantas**  
MSc Informatics – AI (PPGI/UFPB) • R&D Data Scientist — Computer Vision | GenAI

---
© 2025 Lucas G. Dantas — All rights reserved.
