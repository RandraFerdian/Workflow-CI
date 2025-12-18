import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

# --- 1. KONFIGURASI DAGSHUB ---
DAGSHUB_URI = "https://dagshub.com/RandraFerdian/Eksperimen_SML_Randra.mlflow"

# Setup Path (Sing Bener)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "btc_data_preprocessed") 
train_path = os.path.join(DATA_PATH, "train.csv")
test_path = os.path.join(DATA_PATH, "test.csv")

def load_data():
    print("[INFO] Loading data...")
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"[ERROR] Data ora ketemu neng: {DATA_PATH}")
        exit()
    
    X_train = train.drop(columns=['Target'])
    y_train = train['Target']
    X_test = test.drop(columns=['Target'])
    y_test = test['Target']
    return X_train, y_train, X_test, y_test

def train_tuning_advance():
    # SETUP DAGSHUB
    print("[INFO] Nyambungke menyang DagsHub...")
    dagshub.init(repo_owner='RandraFerdian', repo_name='Eksperimen_SML_Randra', mlflow=True)
    mlflow.set_tracking_uri(DAGSHUB_URI)
    
    # ⚠️ REVISI 1: JENENG EKSPERIMEN DISAMAKNE KARO MAIN.YML
    mlflow.set_experiment("Bitcoin_Sniper_FIX_311")

    with mlflow.start_run():
        print("[INFO] Mulai Hyperparameter Tuning...")
        X_train, y_train, X_test, y_test = load_data()

        # --- A. TUNING ---
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10], 
            'min_samples_split': [5, 10]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"[SUCCESS] Best Params: {best_params}")

        # --- B. LOGGING ---
        mlflow.log_params(best_params)
        
        predictions = best_model.predict(X_test)
        
        metrics = {
            "test_accuracy": accuracy_score(y_test, predictions),
            "test_precision": precision_score(y_test, predictions),
            "test_recall": recall_score(y_test, predictions),
            "test_f1": f1_score(y_test, predictions)
        }
        mlflow.log_metrics(metrics)
        print(f"[RESULT] Testing Metrics: {metrics}")

        # --- C. ARTEFAK GAMBAR ---
        plt.figure(figsize=(6,5))
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix (Testing)")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        
        # REVISI 2 & 3: JENENG MODEL & ENVIRONMENT
        # Kita nggunakake 'best_random_forest_final' (padha karo main.yml)
        # Kita nambah 'conda_env' ben Python 3.11 kebaca neng Docker
        print("[INFO] Logging model menyang MLflow...")
        mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path="best_random_forest_final", 
            conda_env="conda.yaml"
        )
        
        # Resik-resik
        if os.path.exists("confusion_matrix.png"): os.remove("confusion_matrix.png")
        
        print("[DONE] SUKSES! Model Python 3.11 wis munggah DagsHub.")

if __name__ == "__main__":
    train_tuning_advance()