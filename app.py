import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template

# Set up environment variables to disable oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from utils import extract_signal_features, load_sound_file

app = Flask(__name__)

# Constants for feature extraction
N_MELS = 64
FRAMES = 5
N_FFT = 1024

# Global variables to cache models and thresholds
models = {}
thresholds = {}

def load_system():
    """Load both models and thresholds into memory on startup"""
    global models, thresholds
    
    # Configure Keras to allow custom objects without issues
    # Using MSE loss during inference doesn't actually matter because we manually compute reconstruction error
    # but we load the keras models here.
    
    saved_dir = "saved_models"
    
    for machine in ["idmt", "mimii"]:
        model_path = os.path.join(saved_dir, f"{machine}_model.keras")
        thresh_path = os.path.join(saved_dir, f"{machine}_threshold.json")
        
        if os.path.exists(model_path) and os.path.exists(thresh_path):
            try:
                # Load Threshold
                with open(thresh_path, 'r') as f:
                    data = json.load(f)
                    thresholds[machine] = data["threshold"]
                    
                # Load Model (Add safe mode params if tf is missing custom loss objects)
                models[machine] = tf.keras.models.load_model(model_path, compile=False)
                
                print(f"[+] Successfully loaded {machine} subsystem. Threshold: {thresholds[machine]:.2f}")
            except Exception as e:
                print(f"[-] Error loading {machine} model: {e}")
        else:
            print(f"[-] Wait: Model for {machine} not found in {saved_dir}/. You must run baseline5.py first.")

@app.route('/')
def index():
    # Serve the main HTML page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400
        
    audio_file = request.files['audio']
    machine_type = request.form.get('type')
    
    if not audio_file.filename.endswith('.wav'):
        return jsonify({'error': 'Only .wav files are supported'}), 400
        
    if machine_type not in ['idmt', 'mimii']:
        return jsonify({'error': 'Invalid machine type. Must be idmt (Moteur) or mimii (Pompe).'}), 400
        
    if machine_type not in models or machine_type not in thresholds:
        return jsonify({'error': f'Model {machine_type} is not prepared/trained yet.'}), 500

    # 1. Save uploaded file temporarily
    temp_path = os.path.join("temp", audio_file.filename)
    os.makedirs("temp", exist_ok=True)
    audio_file.save(temp_path)
    
    try:
        # 2. Extract Features
        signal, sr = load_sound_file(temp_path)
        features = extract_signal_features(signal, sr, n_mels=N_MELS, frames=FRAMES, n_fft=N_FFT)
        
        if len(features) == 0:
            return jsonify({'error': 'Audio file too short to analyze'}), 400
            
        # 3. Model Prediction for BOTH models (for cross-checking)
        model = models[machine_type]
        predictions = model.predict(features, verbose=0)
        mse = float(np.mean(np.square(features - predictions)))
        
        # Check the other machine to see if it fits better
        other_machine = 'mimii' if machine_type == 'idmt' else 'idmt'
        model_other = models[other_machine]
        pred_other = model_other.predict(features, verbose=0)
        mse_other = float(np.mean(np.square(features - pred_other)))
        
        threshold = thresholds[machine_type]
        threshold_other = thresholds[other_machine]
        
        # 4. Out-of-Distribution (Wrong Machine) Check
        # If the sound is horribly badly reconstructed by the selected model (e.g. 3x threshold)
        # AND it is much better reconstructed by the OTHER model (or it's just garbage noise),
        # we reject it.
        # Simple heuristic: if MSE is huge, and (MSE_other / threshold_other) < (MSE / threshold)
        is_wrong_machine = False
        
        # If the error is massively above threshold (say 2.5x), it's probably not even the right machine
        if mse > (threshold * 2.5):
            is_wrong_machine = True
            
        if is_wrong_machine:
            return jsonify({
                'status': 'WrongMachine',
                'selected_type': machine_type,
                'score': mse,
                'threshold': threshold
            })
            
        # 5. Verdict
        is_anomaly = mse > threshold
        
        return jsonify({
            'status': 'Anomaly' if is_anomaly else 'Normal',
            'score': mse,
            'threshold': threshold,
            'machine_type': machine_type
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    load_system()
    app.run(debug=True, port=5000)
