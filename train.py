import mlflow
import dvc.api
import pandas as pd
import subprocess
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.tracking import get_experiment_by_name

# --- CONFIGURATION & PARAMETER LOADING ---
MLFLOW_TRACKING_URI = "http://localhost:5001" 
EXPERIMENT_NAME = "Taxi_Fare_Prediction_CT"
DVC_DATA_PATH = 'data/raw/train.csv'
PARAMS_FILE = 'params.yaml'

# Load parameters from params.yaml
try:
    with open(PARAMS_FILE, 'r') as f:
        params = yaml.safe_load(f)['train']
except FileNotFoundError:
    print(f"Error: Parameter file '{PARAMS_FILE}' not found. Please create it.")
    exit()

# Extract parameters
C_HYPERPARAMETER = params.get('C_hyperparameter', 0.5)
RANDOM_SEED = params.get('random_state', 42)
TEST_SIZE = params.get('test_size', 0.2)

# --- MLFLOW SETUP ---
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
# Retrieve the experiment object for logging the final URL
experiment = get_experiment_by_name(EXPERIMENT_NAME)


# --- DATA RETRIEVAL VIA DVC ---
print(f"Retrieving data from DVC: {DVC_DATA_PATH}")

# Ensure the data file tracked by DVC is locally present
try:
    # Use dvc.api to interact with the tracked data
    dvc.api.get_url(DVC_DATA_PATH)
except Exception as e:
    print(f"Error fetching DVC data: {e}. Make sure you ran 'dvc add {DVC_DATA_PATH}' and 'dvc push'.")
    exit()

df = pd.read_csv(DVC_DATA_PATH)

# =========================================================
# DATA PREPARATION & SPLITTING
# =========================================================

# 1. Feature Engineering & Selection 
df['is_long_trip'] = (df['trip_distance'] > 5.0).astype(int)

# 2. Define Features (X) and Target (y)
X = df[['passenger_count', 'trip_distance']]
y = df['is_long_trip']

# 3. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# =========================================================
# MLFLOW TRACKING & TRAINING
# =========================================================
with mlflow.start_run(run_name="dvc-ct-run"):
    
    # 1. Log Hyperparameters (Read directly from the loaded params)
    mlflow.log_param("regularization_C", C_HYPERPARAMETER)
    mlflow.log_param("test_size", TEST_SIZE)
    
    # 2. Train Model
    print("Starting model training...")
    model = LogisticRegression(C=C_HYPERPARAMETER, random_state=RANDOM_SEED).fit(X_train, y_train)
    
    # 3. Evaluate and Log Metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("test_accuracy", accuracy)
    
    # 4. Log Data Version Tag (FIXED: Using subprocess for Git hash)
    try:
        # Get the current Git commit hash
        git_commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).strip().decode('utf-8')
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit_hash = "unknown"
        print("Warning: Could not retrieve Git commit hash. Is 'git' installed and initialized?")

    mlflow.set_tag("data_commit_hash", git_commit_hash)
    
    # 5. Save and Register Model
    # Output artifact folder used by DVC pipeline: 'model/'
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="Production_CT_Model"
    )

    run_id = mlflow.last_active_run().info.run_id

# =========================================================
# FINAL OUTPUT
# =========================================================
print(f"\nTraining Complete. Test Accuracy: {accuracy:.4f}")
print(f"MLflow Run ID: {run_id}")
print(f"View run at: {MLFLOW_TRACKING_URI}/#/experiments/{experiment.experiment_id}/runs/{run_id}")