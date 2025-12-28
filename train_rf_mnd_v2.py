import os
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

DATA_DIR = "generated"
LABELS = ["ALS", "PMA", "PBP", "NORMAL"]

# =============================
# Utilities
# =============================
def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

def severity_score(val):
    return {
        "+":1,"+1":1,
        "++":2,"+2":2,
        "+++":3,"+3":3,
        "+4":4
    }.get(str(val).strip(), 0)

# =============================
# Feature Extraction (CLINICAL)
# =============================
def extract_features(json_path):
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    features = {}

    # ---------- EMG ----------
    emg = data.get("Electromyography", [])
    bulbar_kw = ["oris", "tongue", "genio"]

    bulbar = 0
    limb = 0
    fibs = 0
    psw = 0

    for e in emg:
        m = str(e.get("Muscles", "")).lower()

        if any(k in m for k in bulbar_kw):
            bulbar += 1
        else:
            limb += 1

        fibs += severity_score(e.get("Fibs"))
        psw  += severity_score(e.get("Psw"))

    denervation = int((fibs + psw) > 0)
    emg_normal = int(len(emg) > 0 and denervation == 0)

    features["emg_count"] = len(emg)
    features["emg_bulbar_count"] = bulbar
    features["emg_limb_count"] = limb
    features["emg_denervation"] = denervation
    features["emg_normal"] = emg_normal

    # ---------- MNCS ----------
    mncs = data.get("Motor_Nerve_Conduction_Studies", [])
    amps = [
        safe_float(m.get("Amplitude_mv"))
        for m in mncs
        if not np.isnan(safe_float(m.get("Amplitude_mv")))
    ]

    features["mncs_count"] = len(mncs)
    features["mncs_amp_min"] = min(amps) if amps else 0.0

    # ---------- SNCS ----------
    sncs = data.get("Sensory_Nerve_Conduction_Studies", [])
    features["sncs_count"] = len(sncs)

    return features

# =============================
# Load Dataset (ANTI-BIAS)
# =============================
X, y = [], []

for label in LABELS:
    folder = os.path.join(DATA_DIR, label)
    files = [f for f in os.listdir(folder) if f.endswith(".json")]

    # ⚠️ Control NORMAL dominance
    if label == "NORMAL":
        files = files[:150]

    for fname in files:
        X.append(extract_features(os.path.join(folder, fname)))
        y.append(label)

feature_names = list(X[0].keys())
X_mat = np.array([[x[f] for f in feature_names] for x in X], dtype=float)
X_mat = np.nan_to_num(X_mat)

le = LabelEncoder()
y_enc = le.fit_transform(y)

# =============================
# Train / Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X_mat,
    y_enc,
    test_size=0.2,
    stratify=y_enc,
    random_state=42
)

# =============================
# Train Random Forest (CLINICAL)
# =============================
model = RandomForestClassifier(
    n_estimators=800,
    max_depth=14,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =============================
# Evaluation
# =============================
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# =============================
# Save Model
# =============================
joblib.dump(model, "rf_mnd_model_FINAL.joblib")
joblib.dump(feature_names, "rf_features_FINAL.joblib")
joblib.dump(le, "rf_label_encoder_FINAL.joblib")

print("\n✅ FINAL MND model trained (clinically aligned)")
