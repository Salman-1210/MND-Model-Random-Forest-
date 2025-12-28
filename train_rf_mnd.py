import os
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

DATA_DIR = "generated"

# -----------------------------
# Utility
# -----------------------------
def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

def severity_score(val):
    if val in ["+", "+1"]: return 1
    if val in ["++", "+2"]: return 2
    if val in ["+++", "+3"]: return 3
    if val in ["+4"]: return 4
    return 0

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = {}

    # ===== EMG =====
    emg = data.get("Electromyography", [])
    features["emg_count"] = len(emg)

    bulbar_kw = ["oris", "tongue", "genio"]
    bulbar = 0
    limb = 0
    fibs_sum = 0
    psw_sum = 0

    for e in emg:
        m = str(e.get("Muscles", "")).lower()
        if any(k in m for k in bulbar_kw):
            bulbar += 1
        else:
            limb += 1
        fibs_sum += severity_score(str(e.get("Fibs", "")))
        psw_sum += severity_score(str(e.get("Psw", "")))

    features["emg_bulbar_count"] = bulbar
    features["emg_limb_count"] = limb
    features["emg_fibs_score"] = fibs_sum
    features["emg_psw_score"] = psw_sum

    # ===== MNCS =====
    mncs = data.get("Motor_Nerve_Conduction_Studies", [])
    lat, amp, ncv = [], [], []
    nr_count = 0

    for m in mncs:
        lat.append(safe_float(m.get("Latency_ms")))
        amp.append(safe_float(m.get("Amplitude_mv")))
        ncv.append(safe_float(m.get("NCV_ms")))
        if str(m.get("Latency_ms")).upper().startswith("NR"):
            nr_count += 1

    features["mncs_count"] = len(mncs)
    features["mncs_nr_count"] = nr_count
    features["mncs_latency_mean"] = np.nanmean(lat) if lat else 0
    features["mncs_amplitude_mean"] = np.nanmean(amp) if amp else 0
    features["mncs_ncv_mean"] = np.nanmean(ncv) if ncv else 0

    # ===== SNCS =====
    sncs = data.get("Sensory_Nerve_Conduction_Studies", [])
    lat, amp, ncv = [], [], []
    nr_count = 0

    for s in sncs:
        lat.append(safe_float(s.get("Latency_ms")))
        amp.append(safe_float(s.get("Amplitude_uv")))
        ncv.append(safe_float(s.get("NCV_ms")))
        if str(s.get("Latency_ms")).upper().startswith("NR"):
            nr_count += 1

    features["sncs_count"] = len(sncs)
    features["sncs_nr_count"] = nr_count
    features["sncs_latency_mean"] = np.nanmean(lat) if lat else 0
    features["sncs_amplitude_mean"] = np.nanmean(amp) if amp else 0
    features["sncs_ncv_mean"] = np.nanmean(ncv) if ncv else 0

    return features

# -----------------------------
# Load Dataset (4 classes)
# -----------------------------
X = []
y = []

LABELS = ["ALS", "PMA", "PBP", "NORMAL"]

for label in LABELS:
    folder = os.path.join(DATA_DIR, label)
    for fname in os.listdir(folder):
        if fname.endswith(".json"):
            path = os.path.join(folder, fname)
            X.append(extract_features(path))
            y.append(label)

# Convert to matrix
feature_names = list(X[0].keys())
X_mat = np.array([[x[f] for f in feature_names] for x in X], dtype=float)

# Safety
X_mat = np.nan_to_num(X_mat, nan=0.0)

le = LabelEncoder()
y_enc = le.fit_transform(y)

# -----------------------------
# Train / Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_mat,
    y_enc,
    test_size=0.2,
    stratify=y_enc,
    random_state=42
)

# -----------------------------
# Train Random Forest
# -----------------------------
model =  RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
) 

model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "rf_mnd_model.joblib")
joblib.dump(feature_names, "rf_features.joblib")
joblib.dump(le, "rf_label_encoder.joblib")
print("\n✅ Model training complete with NORMAL class.")