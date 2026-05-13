import os
import pandas as pd
import numpy as np
import pickle
import random

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../icu/data/icu_risk_model.pkl')
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/kafka_streaming_data_explained.csv')
OUT_PATH = os.path.join(os.path.dirname(__file__), '../data/kafka_streaming_data_final.csv')

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data not found at {DATA_PATH}")
        return

    print("Loading explained data...")
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Assign each patient a random admission offset (0 to 30 minutes apart)
    random.seed(42)
    patients = df['subject_id'].unique()
    offsets = {pid: pd.Timedelta(seconds=random.randint(0, 1800)) for pid in patients}

    df['timestamp'] = df.apply(
        lambda row: row['timestamp'] + offsets[row['subject_id']], axis=1
    )

    df = df.sort_values(['subject_id', 'timestamp']).reset_index(drop=True)

    vitals = ['heart_rate', 'spo2', 'systolic_bp', 'diastolic_bp', 'mean_bp']
    smoothed_groups = []
    print("Smoothing trajectories...")
    for pid, group in df.groupby('subject_id'):
        group = group.copy().reset_index(drop=True)
        for v in vitals:
            # EMA smoothing - alpha=0.3 means 70% previous, 30% new
            group[v] = group[v].ewm(alpha=0.3, adjust=False).mean().round(2)
        # Recalculate MAP from smoothed BP values
        group['mean_bp'] = ((group['systolic_bp'] + 2 * group['diastolic_bp']) / 3).round(2)
        # Clip SpO2 to valid range
        group['spo2'] = group['spo2'].clip(upper=100.0)
        smoothed_groups.append(group)

    df_smooth = pd.concat(smoothed_groups).reset_index(drop=True)

    # Show comparison
    p = df_smooth[df_smooth['subject_id']=='SYN_CT_1000']['heart_rate'].values[:10]
    print("Sample HR after smoothing:", p)

    # Re-predict warnings with the smoothed data using the ICU Risk Model
    if not os.path.exists(MODEL_PATH):
        local_model = os.path.join(os.path.dirname(__file__), '../data/icu_risk_model.pkl')
        if os.path.exists(local_model):
            global MODEL_PATH
            MODEL_PATH = local_model
        else:
            print(f"Warning: Model not found at {MODEL_PATH}. Cannot update warning probabilities.")
            df_smooth.to_csv(OUT_PATH, index=False)
            print(f"Smoothed trajectories saved to {OUT_PATH}")
            return

    print("Loading risk model for re-prediction...")
    model = pickle.load(open(MODEL_PATH, 'rb'))
    features = ['heart_rate', 'spo2', 'systolic_bp', 'diastolic_bp', 'mean_bp']
    X_smooth = df_smooth[features]

    df_smooth['warning'] = model.predict(X_smooth)
    df_smooth['warning_prob'] = model.predict_proba(X_smooth)[:, 1].round(3)
    # warning_prob = how confident the model is (0.0 to 1.0)
    # useful for dashboard: show "Risk: 87%" instead of just 0/1

    df_smooth.to_csv(OUT_PATH, index=False)
    print(f"Final smoothed and predicted trajectories saved to {OUT_PATH}")

if __name__ == "__main__":
    main()