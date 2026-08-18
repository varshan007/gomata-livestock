import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

# Configure Logging
log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "train_anomaly.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TrainAnomaly")

def main():
    logger.info("Starting Isolation Forest training for Health Anomaly Detection...")

    # Load Data
    data_path = os.path.join(os.path.dirname(__file__), "../training_data/health_dataset.csv")
    if not os.path.exists(data_path):
        logger.error(f"Training data not found at {data_path}. Please run extract_features.py first.")
        return

    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset with {len(df)} records.")

    # Define feature columns based on script extraction output
    feature_cols = [
        'temp_current', 'temp_6h_avg', 'temp_6h_std', 'temp_6h_slope',
        'hr_current', 'hr_6h_avg', 'hr_6h_std', 'hr_6h_slope',
        'activity_current', 'activity_6h_avg', 'activity_6h_std', 'activity_6h_slope'
    ]

    # Check if features exist
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns in dataset: {missing_cols}")
        return

    # Extract features and labels
    X = df[feature_cols].fillna(0)
    y = df['label'].fillna(0)

    # Split into train/test to evaluate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Set up MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment("Health_Anomaly_Detection_IsolationForest")

    with mlflow.start_run():
        contamination = 0.20 # Based on expected anomaly ratio of around 20%
        clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        
        logger.info(f"Training Isolation Forest with contamination={contamination}...")
        clf.fit(X_train_scaled)

        # Evaluate
        # IsolationForest returns 1 for inliers (normal) and -1 for outliers (anomalies)
        # We need to map it to 0 (normal) and 1 (anomaly) to match our labels
        y_pred_train_raw = clf.predict(X_train_scaled)
        y_pred_train = np.where(y_pred_train_raw == -1, 1, 0)
        
        y_pred_test_raw = clf.predict(X_test_scaled)
        y_pred_test = np.where(y_pred_test_raw == -1, 1, 0)

        # Metrics
        logger.info("Training Classification Report:")
        logger.info("\n" + classification_report(y_train, y_pred_train))
        
        logger.info("Test Classification Report:")
        
        # Determine the string or float key for True/1.0 in classification report
        report_test = classification_report(y_test, y_pred_test, output_dict=True)
        lbl_1 = '1.0' if '1.0' in report_test else '1'
        
        logger.info("\n" + classification_report(y_test, y_pred_test))

        # Log parameters and metrics to MLflow
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("n_estimators", 100)
        
        if lbl_1 in report_test:
            mlflow.log_metric("test_precision", report_test[lbl_1]['precision'])
            mlflow.log_metric("test_recall", report_test[lbl_1]['recall'])
            mlflow.log_metric("test_f1", report_test[lbl_1]['f1-score'])

        # Save model and scaler locally
        models_dir = os.path.join(os.path.dirname(__file__), "../models")
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, "isolation_forest.joblib")
        scaler_path = os.path.join(models_dir, "scaler.joblib")
        
        joblib.dump(clf, model_path)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")

        # Log model to MLflow
        mlflow.sklearn.log_model(clf, "isolation_forest_model")
        
    logger.info("Training completed.")

if __name__ == "__main__":
    main()
