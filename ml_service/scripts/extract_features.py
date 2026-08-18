import os
import sys
import time
import logging
from datetime import timedelta
import numpy as np
import pandas as pd
from pymongo import MongoClient

# Configure Logging
log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "extract_features.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FeatureExtractor")

# Constants
MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/livestock_monitoring")
DB_NAME = "livestock_monitoring"
BATCH_SIZE = 5000
MAX_TELEMETRY_RECORDS = 72
MIN_TELEMETRY_RECORDS = 5
NORMAL_SAMPLE_SIZE = 75000

# Setup MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
telemetry_col = db["devicetelemetries"]
events_col = db["trainingevents"]

def get_event_time(event):
    """
    Resolve correct timestamp field for training event.
    Priority order:
    1. createdAt
    2. timestamp
    3. recordedAt
    """
    t = event.get("createdAt") or event.get("timestamp") or event.get("recordedAt")
    if not t:
        return None
    return t

def compute_slope(series):
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series))
    try:
        slope, _ = np.polyfit(x, series, 1)
        return float(slope)
    except Exception:
        return 0.0

def extract_features_for_event(event):
    animal_id = event.get('animalId')
    created_at = get_event_time(event)
    
    if not animal_id or not created_at:
        return None

    # Fetch telemetry
    telemetry_cursor = telemetry_col.find({
        "animalId": animal_id
    }).sort("timestamp", -1).limit(MAX_TELEMETRY_RECORDS)
    
    telemetry_list = list(telemetry_cursor)
    
    if len(telemetry_list) < MIN_TELEMETRY_RECORDS:
        return None

    # Parse telemetry points
    temps = []
    hrs = []
    activities = []

    for t in telemetry_list:
        temps.append(t.get('temperature', 39.0))
        hrs.append(t.get('heartRate', 60))
        activities.append(float(t.get('activity', 0)))

    # Reverse to chronological order for slope calculation
    temps.reverse()
    hrs.reverse()
    activities.reverse()
    
    # Base Features
    temp_np = np.array(temps)
    hr_np = np.array(hrs)
    act_np = np.array(activities)

    # Anomaly status label from training event
    metadata = event.get('metadata')
    is_anomaly = 1.0 if metadata else 0.0
    event_type = event.get('eventType', 'normal')

    # Convert intensity from string if present
    intensity_raw = metadata.get('intensity', 0.0) if metadata else 0.0
    intensity_val = 0.0
    if isinstance(intensity_raw, str):
        if 'severe' in intensity_raw: intensity_val = 1.0
        elif 'moderate' in intensity_raw: intensity_val = 0.5
        elif 'mild' in intensity_raw: intensity_val = 0.2
    else:
        intensity_val = float(intensity_raw)

    feature_row = {
        'animal_id': str(animal_id),
        'timestamp': created_at.isoformat() if created_at else None,

        # Temperature
        'temp_current': float(temps[-1]),
        'temp_6h_avg': float(np.mean(temp_np)),
        'temp_6h_std': float(np.std(temp_np)),
        'temp_6h_slope': compute_slope(temp_np),
        'temp_max_6h': float(np.max(temp_np)),
        'temp_min_6h': float(np.min(temp_np)),
        'temp_range_6h': float(np.max(temp_np) - np.min(temp_np)),

        # Heart Rate
        'hr_current': float(hrs[-1]),
        'hr_6h_avg': float(np.mean(hr_np)),
        'hr_6h_std': float(np.std(hr_np)),
        'hr_6h_slope': compute_slope(hr_np),

        # Activity
        'activity_current': float(activities[-1]),
        'activity_6h_avg': float(np.mean(act_np)),
        'activity_6h_std': float(np.std(act_np)),
        'activity_6h_slope': compute_slope(act_np),

        # Episode Metadata Features (0 if normal)
        'intensity': intensity_val,
        'phase': metadata.get('phase', 'normal') if metadata else 'normal',
        'correlation_strength': float(metadata.get('correlationStrength', 0.0)) if metadata else 0.0,

        # Labels
        'label': is_anomaly,
        'event_type': event_type
    }
    
    return feature_row

def process():
    logger.info("Starting production ML feature extraction pipeline...")

    # Output paths
    output_dir = os.path.join(os.path.dirname(__file__), "../training_data")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "health_dataset.csv")
    parquet_path = os.path.join(output_dir, "health_dataset.parquet")

    # 1. Pipeline Query Setup
    logger.info("Setting up query for Training Events...")
    base_query = {"source": {"$in": ["simulation", "simulation_v2"]}}

    # Fetch Anomalies
    anomaly_query = {**base_query, "metadata": {"$exists": True}}
    total_anomalies_db = events_col.count_documents(anomaly_query)
    logger.info(f"Connecting to MongoDB... Target Anomalies: {total_anomalies_db}")

    anomaly_cursor = events_col.find(anomaly_query)

    # Fetch Normals (Random Subsample using Aggregation)
    logger.info(f"Targeting Normal sample size limit: {NORMAL_SAMPLE_SIZE}")
    normal_pipeline = [
        {"$match": {**base_query, "metadata": {"$exists": False}}},
        {"$sample": {"size": NORMAL_SAMPLE_SIZE}}
    ]
    normal_docs = list(events_col.aggregate(normal_pipeline))
    logger.info(f"Successfully sampled {len(normal_docs)} Normal events.")

    # 2. Extract Features
    all_events = list(anomaly_cursor) + normal_docs
    np.random.shuffle(all_events)  # Shuffle before processing
    
    total_events = len(all_events)
    logger.info(f"Total events queued for extraction: {total_events}")

    dataset = []
    processed_count = 0
    success_count = 0
    skipped_count = 0

    # Batch Process
    for event in all_events:
        try:
            feats = extract_features_for_event(event)
            if feats:
                dataset.append(feats)
                success_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            logger.debug(f"Error processing event {event.get('_id')}: {e}")
            skipped_count += 1

        processed_count += 1
        
        # Log Progress
        if processed_count % BATCH_SIZE == 0:
            yield_pct = (success_count / processed_count) * 100 if processed_count > 0 else 0
            logger.info(f"Processed Batch: {processed_count}/{total_events} | Valid: {success_count} | Skipped: {skipped_count} | Yield: {yield_pct:.1f}%")

    # 3. Validation Logging & Export
    yield_percentage = (success_count / total_events) * 100 if total_events > 0 else 0
    
    logger.info("=" * 40)
    logger.info("EXTRACTION SUMMARY")
    logger.info("=" * 40)
    logger.info(f"Total events processed: {total_events}")
    logger.info(f"Valid records: {success_count}")
    logger.info(f"Skipped records: {skipped_count}")
    logger.info(f"Extraction yield: {yield_percentage:.1f}%")
    logger.info("=" * 40)

    if success_count == 0:
        logger.error("No valid features extracted. Exiting.")
        return

    df = pd.DataFrame(dataset)
    
    anomaly_count = len(df[df['label'] == 1.0])
    normal_count = len(df[df['label'] == 0.0])
    anomaly_ratio = (anomaly_count / success_count) * 100

    logger.info(f"Final Dataset Size: {success_count} records")
    logger.info(f"Class Distribution -> Anomalies: {anomaly_count} ({anomaly_ratio:.2f}%) | Normals: {normal_count}")

    df.to_csv(csv_path, index=False)
    logger.info(f"Exported CSV: {csv_path}")

    # Convert non-string object types to string for parquet compatibility
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].astype(str)

    df.to_parquet(parquet_path, engine='fastparquet')
    logger.info(f"Exported Parquet: {parquet_path}")

if __name__ == "__main__":
    process()