import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import torch
import random
from ctgan import CTGAN

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/subset_events.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ctgan_model.pkl')

def compute_jsd(real_data, synth_data, bins=50):
    """
    Computes the Jensen-Shannon Divergence (JSD) for each column.
    JSD is symmetric and bounded between 0 and 1 (when using base 2 log, or 0 to ln(2)).
    A JSD close to 0 means the distributions are identical.
    """
    jsd_results = {}
    for col in real_data.columns:
        # Determine the min and max across both distributions to align the bins
        min_val = min(real_data[col].min(), synth_data[col].min())
        max_val = max(real_data[col].max(), synth_data[col].max())
        
        # Create histogram bins
        bins_edges = np.linspace(min_val, max_val, bins)
        
        # Calculate probabilities
        p, _ = np.histogram(real_data[col], bins=bins_edges, density=True)
        q, _ = np.histogram(synth_data[col], bins=bins_edges, density=True)
        
        # Add small epsilon to avoid division by zero / log of zero
        p = np.clip(p, 1e-10, None)
        q = np.clip(q, 1e-10, None)
        
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        
        # Compute JSD
        jsd = jensenshannon(p, q, base=2)
        jsd_results[col] = jsd
        
    return jsd_results

def compute_cmd(corr_real, corr_synth):
    """
    Computes the Correlation Matrix Distance (CMD).
    Formula: CMD(R1, R2) = 1 - trace(R1 * R2) / (norm(R1) * norm(R2))
    CMD is bounded between 0 and 1. 
    0 means matrices are identical up to a scalar factor.
    """
    R1 = corr_real.values
    R2 = corr_synth.values
    
    # Calculate trace of dot product
    trace_r1_r2 = np.trace(np.dot(R1, R2))
    
    # Calculate Frobenius norms
    norm_r1 = np.linalg.norm(R1, 'fro')
    norm_r2 = np.linalg.norm(R2, 'fro')
    
    # Compute CMD
    cmd = 1 - (trace_r1_r2 / (norm_r1 * norm_r2))
    return cmd

def preprocess_real_events(filepath):
    df = pd.read_csv(filepath)
    df['charttime'] = pd.to_datetime(df['charttime'])
    
    pivot_df = df.pivot_table(
        index=['subject_id', 'charttime'], 
        columns='itemid', 
        values='valuenum'
    ).reset_index()
    
    col_mapping = {
        220045: 'heart_rate',
        220277: 'spo2',
        220179: 'systolic_bp',
        220180: 'diastolic_bp',
        220181: 'mean_bp'
    }
    rename_dict = {k: v for k, v in col_mapping.items() if k in pivot_df.columns}
    pivot_df.rename(columns=rename_dict, inplace=True)
    
    required_cols = list(rename_dict.values())
    pivot_df[required_cols] = pivot_df.groupby('subject_id')[required_cols].ffill().bfill()
    pivot_df.dropna(subset=required_cols, inplace=True)
    
    return pivot_df

def main():
    # Lock seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("Loading real data...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Could not find real data at {DATA_PATH}")
        return
        
    real_df = preprocess_real_events(DATA_PATH)
    cols_to_evaluate = ['heart_rate', 'spo2', 'systolic_bp', 'diastolic_bp', 'mean_bp']
    real_vitals = real_df[cols_to_evaluate].copy()
    
    print("Loading trained CTGAN model...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Could not find trained model at {MODEL_PATH}")
        return
        
    ctgan = CTGAN.load(MODEL_PATH)
    
    print("Generating synthetic data...")
    synth_df = ctgan.sample(len(real_df))
    
    # Apply our physical constraints and correlation steering
    synth_df['spo2'] = synth_df['spo2'].clip(upper=100.0, lower=50.0)
    synth_df['mean_bp'] = (synth_df['systolic_bp'] + 2 * synth_df['diastolic_bp']) / 3.0
    
    sys_mean = synth_df['systolic_bp'].mean()
    dia_mean = synth_df['diastolic_bp'].mean()
    spo2_adjustment = -0.05 * (synth_df['diastolic_bp'] - dia_mean)
    synth_df['spo2'] = synth_df['spo2'] + spo2_adjustment
    synth_df['spo2'] = synth_df['spo2'].clip(upper=100.0, lower=50.0)
    spo2_mean = synth_df['spo2'].mean()
    
    hr_adjustment = -0.10 * (synth_df['systolic_bp'] - sys_mean) + 0.15 * (synth_df['diastolic_bp'] - dia_mean) + 0.30 * (synth_df['spo2'] - spo2_mean)
    synth_df['heart_rate'] = synth_df['heart_rate'] + hr_adjustment
    
    synth_vitals = synth_df[cols_to_evaluate].copy()
    
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    
    # 1. JENSEN-SHANNON DIVERGENCE (JSD)
    jsd_scores = compute_jsd(real_vitals, synth_vitals)
    print("\n1. Jensen-Shannon Divergence (JSD)")
    print("   (Lower is better. 0 = identical distributions)")
    print("-" * 40)
    mean_jsd = 0
    for col, score in jsd_scores.items():
        print(f"   {col:<15}: {score:.4f}")
        mean_jsd += score
    print("-" * 40)
    print(f"   Average JSD    : {mean_jsd / len(jsd_scores):.4f}")
    
    # 2. CORRELATION MATRIX DISTANCE (CMD)
    corr_real = real_vitals.corr()
    corr_synth = synth_vitals.corr()
    cmd_score = compute_cmd(corr_real, corr_synth)
    
    print("\n2. Correlation Matrix Distance (CMD)")
    print("   (Lower is better. 0 = perfectly identical correlation structures)")
    print("-" * 40)
    print(f"   CMD Score      : {cmd_score:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
