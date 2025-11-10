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
│   └── tasks/            # Specific ML algorithms (KNN, SVM, ANN, etc.)
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 🧩 Features
- Synthetic dataset generation — bivariate Gaussian distributions
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


### ✅ Supervised Models
| Model | Notebook | Accuracy | Precision | Recall | F1 |
|:--|:--|--:|--:|--:|--:|
| ⭐️ **Artificial Neural Network (ANN)** | [ann.ipynb](./notebooks/21-ann.ipynb) | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Logistic Regression** | [logreg.ipynb](./notebooks/20-logreg.ipynb) | **0.9833** | **0.9677** | **1.0000** | **0.9836** |
| **SVM** | [svm.ipynb](./notebooks/22-svm.ipynb) | **0.9833** | **0.9677** | **1.0000** | **0.9836** |
| **K-NN (best k = 3)** | [knn.ipynb](./notebooks/24-knn.ipynb) | **0.9833** | **0.9677** | **1.0000** | **0.9836** |
| **Decision Tree** | [tree.ipynb](./notebooks/23-tree.ipynb) | **0.9500** | **0.9655** | **0.9333** | **0.9492** |

> *Notes:* Metrics refer to the **test split (20%)**. Each notebook includes the confusion matrix and decision boundary visualization.

---

### 🌀 Unsupervised Models
| Model | Notebook | Primary Metric(s) | Value |
|:--|:--|:--|:--|
| **K-means** | [kmeans.ipynb](./notebooks/10-kmeans.ipynb) | Adjusted Rand Index (ARI) | **0.9342** |
| **FFuzzy C-Means** | [fcm.ipynb](./notebooks/11-fcm.ipynb) | Centers / Iterations | Centers ≈ `[[60.07, 30.89], [24.51, 9.80]]`; Iter = **15** |

> *Notes:* K-means includes the confusion matrix and decision region visualization.  
> FCM includes fuzzy membership heatmaps.

---

## 👨‍💻 Author
**Lucas G. Dantas**  
MSc Informatics – AI (PPGI/UFPB) • R&D Data Scientist — Computer Vision | GenAI

---
© 2025 Lucas G. Dantas — All rights reserved.
