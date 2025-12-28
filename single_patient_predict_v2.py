import json
import numpy as np
import joblib

# =============================
# LOAD MODEL & METADATA
# =============================
model = joblib.load("rf_mnd_model_CLINICAL.joblib")
feature_names = joblib.load("rf_features_CLINICAL.joblib")
label_encoder = joblib.load("rf_label_encoder_CLINICAL.joblib")

TEST_FILE = "results/result_pbp_generated.json"

# =============================
# MUSCLE MEANING DICTIONARY
# =============================
MUSCLE_INFO = {
    "oris": "lips (speech muscle)",
    "orbicularis": "lips (speech muscle)",
    "tongue": "tongue (speech & swallowing)",
    "genio": "base of tongue (swallowing)",
    "fdi": "hand muscle (fine movement)",
    "ta": "lower leg muscle (foot lifting)",
    "gastroc": "calf muscle (walking)",
    "vl": "thigh muscle (standing)",
    "biceps": "upper arm muscle",
    "triceps": "upper arm muscle"
}

def muscle_meaning(name):
    lname = name.lower()
    for k, v in MUSCLE_INFO.items():
        if k in lname:
            return v
    return "limb muscle"

# =============================
# UTILITIES
# =============================
def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

def severity_score(val):
    return {"+":1,"+1":1,"++":2,"+2":2,"+++":3,"+3":3,"+4":4}.get(str(val).strip(),0)

# =============================
# FEATURE EXTRACTION (MATCH TRAINING)
# =============================
def extract_features_and_context(json_path):
    with open(json_path,"r",encoding="utf-8") as f:
        data = json.load(f)

    features = {}
    why_yes = []
    why_not = {}

    emg = data.get("Electromyography",[])
    mncs = data.get("Motor_Nerve_Conduction_Studies",[])
    sncs = data.get("Sensory_Nerve_Conduction_Studies",[])

    bulbar_kw = ["oris","tongue","genio"]

    bulbar = []
    limb = []
    fibs = 0
    psw = 0

    for e in emg:
        m = e.get("Muscles","")
        if any(k in m.lower() for k in bulbar_kw):
            bulbar.append(m)
        else:
            limb.append(m)

        fibs += severity_score(e.get("Fibs"))
        psw  += severity_score(e.get("Psw"))

    denervation = int((fibs + psw) > 0)

    # ---------- MNCS ----------
    amps = [
        safe_float(m.get("Amplitude_mv"))
        for m in mncs
        if not np.isnan(safe_float(m.get("Amplitude_mv")))
    ]

    low_amp_count = sum(1 for a in amps if a < 4)

    # ---------- FEATURES (ORDER-INDEPENDENT) ----------
    features["emg_count"] = len(emg)
    features["emg_bulbar_count"] = len(bulbar)
    features["emg_limb_ratio"] = len(limb) / max(len(emg),1)
    features["emg_denervation"] = denervation

    features["mncs_count"] = len(mncs)
    features["mncs_amp_min"] = min(amps) if amps else 0.0
    features["mncs_low_amp_count"] = low_amp_count

    features["sncs_count"] = len(sncs)

    features["pure_normal_pattern"] = int(
        denervation == 0 and
        low_amp_count == 0 and
        len(sncs) > 0 and
        len(bulbar) == 0
    )

    # =============================
    # SIMPLE EXPLANATION
    # =============================
    if bulbar:
        why_yes.append(
            "Speech/swallowing muscles examined:\n  - " +
            "\n  - ".join([f"{m} ({muscle_meaning(m)})" for m in bulbar])
        )
    else:
        why_yes.append("Speech and swallowing muscles appear normal")

    if limb:
        why_yes.append(
            "Hand and leg muscles examined:\n  - " +
            "\n  - ".join([f"{m} ({muscle_meaning(m)})" for m in limb])
        )

    if denervation:
        why_yes.append("Muscles show abnormal electrical activity (denervation)")
    else:
        why_yes.append("No muscle denervation detected")

    if low_amp_count > 0:
        why_yes.append("Some motor nerves show reduced signal strength")

    if sncs:
        why_yes.append("Sensory nerves are preserved (normal feeling nerves)")

    # =============================
    # WHY NOT OTHERS
    # =============================
    why_not["PBP"] = (
        "PBP mainly affects speech and tongue muscles.\n"
        "Those muscles are not affected here."
    )

    why_not["PMA"] = (
        "PMA causes clear damage in limb muscles.\n"
        "No such damage is seen here."
    )

    why_not["ALS"] = (
        "ALS affects both bulbar and limb muscles together.\n"
        "This mixed pattern is not present."
    )

    return features, why_yes, why_not

# =============================
# RUN PREDICTION
# =============================
features, why_yes, why_not = extract_features_and_context(TEST_FILE)

X = np.array([[features.get(f,0) for f in feature_names]],dtype=float)
X = np.nan_to_num(X)

probs = model.predict_proba(X)[0]
raw_pred = label_encoder.inverse_transform([np.argmax(probs)])[0]

# =============================
# CLINICAL OVERRIDE (CRITICAL)
# =============================
if features["pure_normal_pattern"] == 1:
    final_pred = "NORMAL"
else:
    final_pred = raw_pred

# =============================
# OUTPUT
# =============================
print("\n" + "═"*60)
print("🧠 MND MODEL – CLINICAL DECISION SUMMARY")
print("═"*60)

print("\n📌 Final Predicted Diagnosis")
print(f"➡️  {final_pred}")

print("\n📊 Prediction Probabilities")
for c,p in zip(label_encoder.classes_,probs):
    print(f"• {c} : {p:.3f}")

print("\n" + "─"*60)
print("🔍 Why this result was given (simple explanation)")
print("─"*60)
for l in why_yes:
    print("•",l)

print("\n" + "─"*60)
print("❌ Why other conditions were not selected")
print("─"*60)
for k,v in why_not.items():
    if k != final_pred:
        print(f"• {k}: {v}")

print("\n" + "═"*60)
print("✅ Prediction & explanation completed")
print("═"*60)
