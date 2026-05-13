import os
import pandas as pd
import pickle
import shap
import numpy as np
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../icu/data/icu_risk_model.pkl')
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/kafka_streaming_data.csv')
OUT_PATH = os.path.join(os.path.dirname(__file__), '../data/kafka_streaming_data_explained.csv')

def main():
    global MODEL_PATH
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        # Fallback to local data dir if the upper one is missing
        local_model = os.path.join(os.path.dirname(__file__), '../data/icu_risk_model.pkl')
        if os.path.exists(local_model):
            MODEL_PATH = local_model
            print(f"Using fallback model at {MODEL_PATH}")
        else:
            return
            
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data not found at {DATA_PATH}")
        return

    print("Loading model and data...")
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    features = ['heart_rate', 'spo2', 'systolic_bp', 'diastolic_bp', 'mean_bp']
    X = df[features]

    print("Calculating SHAP values...")
    explainer = shap.Explainer(model, X)
    shap_vals = explainer(X).values
    
    # Handle classifier SHAP output which is often 3D (samples, features, classes)
    if len(shap_vals.shape) == 3:
        shap_vals = shap_vals[:, :, 1]

    # For each warning row, find top contributing vital
    df['top_trigger'] = ''
    df['trigger_value'] = 0.0

    warning_idx = df[df['warning'] == 1].index
    for i in warning_idx:
        vals = shap_vals[i]
        to
        p_feature_idx = np.argmax(np.abs(vals))
        df.loc[i, 'top_trigger'] = features[top_feature_idx]
        df.loc[i, 'trigger_value'] = X.iloc[i][features[top_feature_idx]]

    # Save enriched file
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved with SHAP trigger columns to {OUT_PATH}")
    
    if len(warning_idx) > 0:
        print("\nTop triggers for warnings:")
        print(df[df['warning']==1]['top_trigger'].value_counts())
    else:
        print("No warnings found in the data.")

if __name__ == "__main__":
    main()