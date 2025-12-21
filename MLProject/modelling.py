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

# Setup Path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "btc_data_preprocessed") 
train_path = os.path.join(DATA_PATH, "train.csv")
test_path = os.path.join(DATA_PATH, "test.csv")

# Custom Env (Tetep dinggo ben Python 3.11 aman)
custom_env = {
    'name': 'bitcoin-env-311',
    'channels': ['defaults'],
    'dependencies': [
        'python=3.11',
        'pip',
        {
            'pip': [
                'mlflow==2.17.2',
                'scikit-learn',
                'pandas',
                'numpy',
                'matplotlib',
                'seaborn',
                'dagshub',
                'virtualenv'
            ]
        }
    ]
}

def load_data():
    # ... (Isi fungsi load_data tetep padha, ora usah diowahi) ...
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        X_train = train.drop(columns=['Target'])
        y_train = train['Target']
        X_test = test.drop(columns=['Target'])
        y_test = test['Target']
        return X_train, y_train, X_test, y_test
    except FileNotFoundError:
        print("Error loading data")
        exit()

def train_tuning_advance():
    print("[INFO] Nyambungke menyang DagsHub...")
    dagshub.init(repo_owner='RandraFerdian', repo_name='Eksperimen_SML_Randra', mlflow=True)
    mlflow.set_tracking_uri(DAGSHUB_URI)
    mlflow.set_experiment("Bitcoin_Ultima_Fix_311")

    # 🔥 REVISI 1: TAMBAH AUTOLOG NENG KENE 🔥
    # Iki bakal otomatis nyathet kabeh parameter GridSearchCV lan model
    mlflow.autolog()

    with mlflow.start_run():
        print("[INFO] Mulai Hyperparameter Tuning...")
        X_train, y_train, X_test, y_test = load_data()

        # --- TUNING ---
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10], 
            'min_samples_split': [5, 10]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        # Kita ora perlu 'mlflow.log_params' maneh amarga wis ana autolog()

        # --- EVALUASI ---
        predictions = best_model.predict(X_test)
        metrics = {
            "test_accuracy": accuracy_score(y_test, predictions),
            "test_precision": precision_score(y_test, predictions),
            "test_recall": recall_score(y_test, predictions),
            "test_f1": f1_score(y_test, predictions)
        }
        mlflow.log_metrics(metrics) # Metrics tambahan tetep oleh
        print(f"[RESULT] Testing Metrics: {metrics}")

        # --- ARTEFAK ---
        # ... (Kode nggawe confusion matrix tetep padha) ...
        
        # ⚠️ LOGGING MODEL (PENTING: Tetep manual ben iso nglebokne custom_env)
        print("[INFO] Logging model karo Python 3.11 Force...")
        mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path="best_random_forest_final", 
            conda_env=custom_env 
        )
        
        run_id = mlflow.active_run().info.run_id
        print(f"RUN_ID_OUTPUT: {run_id}") # Tetep perlu nggo CI/CD
        
        print("[DONE] SUKSES!")

if __name__ == "__main__":
    train_tuning_advance()