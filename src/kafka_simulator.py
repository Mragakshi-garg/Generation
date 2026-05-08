import sys
import os
import time
import json
import random
import datetime
import pandas as pd
from ctgan import CTGAN
from kafka import KafkaProducer

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ctgan_model.pkl')
KAFKA_TOPIC = 'mimic_synthetic_vitals'
KAFKA_BROKER = 'localhost:9092'
STREAM_HZ = 6.0

class SyntheticPatient:
    def __init__(self, subject_id):
        self.subject_id = f"SYN_CT_{subject_id}"
        # Start time
        self.current_time = datetime.datetime.now()
        
        # Base vitals - will be populated by CTGAN on first run
        self.last_vitals = None

def load_model():
    if os.path.exists(MODEL_PATH):
        print(f"Loading trained CTGAN model from {MODEL_PATH}...")
        model = CTGAN.load(MODEL_PATH)
        return model
    else:
        print(f"ERROR: No trained model found at {MODEL_PATH}. Please train the model first.")
        return None

def start_simulation(dry_run=False, max_steps=None):
    model = load_model()
    if model is None:
        return
    
    producer = None
    if not dry_run:
        try:
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BROKER],
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            print(f"Connected to Kafka broker at {KAFKA_BROKER}")
        except Exception as e:
            print(f"Failed to connect to Kafka: {e}")
            print("Falling back to dry-run mode.")
            dry_run = True

    print(f"Starting CTGAN Synthesis Simulation at {STREAM_HZ}Hz...")
    
    # Maintain a pool of 50 synthetic patients in the ICU
    num_patients = 50
    patients = [SyntheticPatient(1000 + i) for i in range(num_patients)]
    
    steps = 0
    try:
        while True:
            if max_steps and steps >= max_steps:
                break
                
            # Ask CTGAN to generate a batch of observations for our patients
            # Note: CTGAN treats each row as independent. For temporal smoothness, 
            # we blend the CTGAN output with the previous state, or just use it directly.
            # Here we just use the raw output to show the true CTGAN distribution,
            # but we apply small micro-variance similar to the GVAE script.
            
            # Generate N rows
            synth_data = model.sample(num_patients)
            
            for i, p in enumerate(patients):
                row = synth_data.iloc[i]
                
                hr = row['heart_rate']
                spo2 = row['spo2']
                sys = row['systolic_bp']
                dia = row['diastolic_bp']
                
                # If this is not the first step, we optionally blend it so it's not wildly jumping every second.
                # CTGAN doesn't natively do time-series smoothing.
                if p.last_vitals is not None:
                    # Blend 80% old, 20% new to simulate continuous observation, while drifting towards CTGAN's sample
                    hr = 0.8 * p.last_vitals['hr'] + 0.2 * hr
                    spo2 = 0.8 * p.last_vitals['spo2'] + 0.2 * spo2
                    sys = 0.8 * p.last_vitals['sys'] + 0.2 * sys
                    dia = 0.8 * p.last_vitals['dia'] + 0.2 * dia
                
                # Add micro-variance
                if random.random() < 0.05:
                    hr += random.choice([-20.0, 30.0]) # Occasional critical spike/drop
                    spo2 -= random.uniform(2.0, 8.0)
                    sys -= random.uniform(10.0, 30.0)
                    dia -= random.uniform(5.0, 15.0)
                else:
                    hr += random.uniform(-2.0, 2.0)
                    spo2 += random.uniform(-0.5, 0.5)
                    sys += random.uniform(-2.0, 2.0)
                    dia += random.uniform(-2.0, 2.0)
                    
                spo2 = min(100.0, max(70.0, float(spo2)))
                mean_bp = (sys + 2 * dia) / 3.0
                
                p.last_vitals = {'hr': hr, 'spo2': spo2, 'sys': sys, 'dia': dia}
                
                # Determine warning status based on clinical risk thresholds
                warning = 1 if (hr > 100 or hr < 50 or spo2 < 90 or sys < 90 or sys > 160 or dia < 60 or dia > 100) else 0
                    
                record = {
                    "subject_id": p.subject_id,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "heart_rate": round(float(hr), 2),
                    "spo2": round(float(spo2), 2),
                    "systolic_bp": round(float(sys), 2),
                    "diastolic_bp": round(float(dia), 2),
                    "mean_bp": round(float(mean_bp), 2),
                    "warning": warning,
                    "is_synthetic": True,
                    "model": "CTGAN"
                }
                
                if dry_run:
                    print(f"[Simulation] JSON emitted: {json.dumps(record)}")
                else:
                    producer.send(KAFKA_TOPIC, value=record)
                    
            time.sleep(1.0)
            steps += 1
            
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        if producer:
            producer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run CTGAN Kafka Simulator")
    parser.add_argument("--dry-run", action="store_true", help="Print to console instead of Kafka")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum steps to simulate")
    args = parser.parse_args()
    
    start_simulation(dry_run=args.dry_run, max_steps=args.max_steps)
