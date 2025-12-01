"""
predict_single_simple.py

Simple standalone tester for flood detection model.
Loads the trained model, scaler, and encoder — then predicts flood status for a single sample.

Steps:
1️⃣ Make sure you’ve trained the model using Train.py (so ./outputs/ folder exists)
2️⃣ Run this file:  python predict_single_simple.py
"""

import joblib
import numpy as np

# ============================================================
# === 1️⃣ Load your trained artifacts =========================
# ============================================================
MODEL_PATH = "./outputs/best_model.joblib"
SCALER_PATH = "./outputs/scaler.joblib"
ENCODER_PATH = "./outputs/label_encoder.joblib"

# Load saved model, scaler, and label encoder
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# ============================================================
# === 2️⃣ Take input from user or set manually ===============
# ============================================================

# Option 1: Enter manually (hardcoded for testing)
# sample = {'water_level': 5.2, 'rain': 3600, 'soil_moisture': 3700}

# Option 2: Ask user interactively
sample = {}
sample['water_level'] = float(input("Enter water level (0–20): "))
sample['rain'] = float(input("Enter rain sensor value (0–4095): "))
sample['soil_moisture'] = float(input("Enter soil moisture (0–4095): "))

# Convert to array
X = np.array([[sample['water_level'], sample['rain'], sample['soil_moisture']]])

# ============================================================
# === 3️⃣ Preprocess input ===================================
# ============================================================
X_scaled = scaler.transform(X)

# ============================================================
# === 4️⃣ Make prediction =====================================
# ============================================================
y_pred = model.predict(X_scaled)
pred_label = label_encoder.inverse_transform(y_pred)[0]

# Get probability (if supported)
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X_scaled)[0]
    class_names = list(label_encoder.classes_)
    print("\n📊 Prediction probabilities:")
    for name, p in zip(class_names, proba):
        print(f"   {name:<10}: {p:.4f}")
else:
    print("\n⚠️  This model does not support probability output.")

# ============================================================
# === 5️⃣ Display the result =================================
# ============================================================
print("\n============================")
print("🌊 FLOOD DETECTION RESULT 🌊")
print("============================")
print(f"Water Level     : {sample['water_level']}")
print(f"Rain Intensity  : {sample['rain']}")
print(f"Soil Moisture   : {sample['soil_moisture']}")
print(f"Predicted Class : ✅ {pred_label.upper()}")
print("============================\n")
