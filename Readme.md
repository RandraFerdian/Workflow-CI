# 🎯 Bitcoin Sniper - End-to-End ML Pipeline & CI/CD

Proyek ini adalah implementasi **Machine Learning Operations (MLOps)** lengkap untuk prediksi pergerakan harga Bitcoin. Proyek ini mencakup siklus pengembangan model dari training, hyperparameter tuning, tracking eksperimen, hingga otomatisasi deployment (CI/CD) menggunakan Docker.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![MLflow](https://img.shields.io/badge/MLflow-Managed-blue?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Container-blue?logo=docker)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-Active-green?logo=github-actions)

## 📋 Fitur Utama

- **Advanced Modeling:** Menggunakan `RandomForestClassifier` dengan Hyperparameter Tuning (`GridSearchCV`) untuk mencari parameter terbaik.
- **Experiment Tracking:** Integrasi penuh dengan **MLflow** dan **DagsHub** untuk mencatat metrik (Accuracy, Precision, Recall, F1), parameter, dan artefak (Confusion Matrix).
- **Environment Locking:** Menggunakan Python 3.11 yang dikunci secara eksplisit di dalam kode untuk menjamin konsistensi antara environment training dan produksi Docker.
- **Automated CI/CD:** Pipeline otomatis menggunakan **GitHub Actions** dengan metode _Direct Handover_ untuk stabilitas deployment.

## 📂 Struktur Proyek

```plaintext
Workflow-CI/
├── .github/
│   └── workflows/
│       └── main.yml        # Konfigurasi CI/CD Pipeline (GitHub Actions)
├── MLProject/
│   ├── btc_data_preprocessed/
│   │   ├── train.csv       # Data Training
│   │   └── test.csv        # Data Testing
│   ├── conda.yaml          # Definisi Environment Conda
│   ├── MLProject           # File Deskripsi Proyek MLflow
│   ├── modelling.py        # Script Utama Training & Logging
│   └── link_docker.txt     # Tautan ke Docker Hub Image
└── README.md               # Dokumentasi Proyek
```
