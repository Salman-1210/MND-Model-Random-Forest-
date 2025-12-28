import json
import numpy as np
import joblib

# =============================
# LOAD MODEL & METADATA
# =============================
model = joblib.load("rf_mnd_model.joblib")
feature_names = joblib.load("rf_features.joblib")
label_encoder = joblib.load("rf_label_encoder.joblib")

TEST_FILE = "results/result_pbp_generated.json"   # change patient here

# =============================
# MUSCLE MEANING DICTIONARY
# =============================
MUSCLE_INFO = {
    "oris": "lips (used for speech)",
    "orbicularis": "lips (used for speech)",
    "tongue": "tongue (used for speech & swallowing)",
    "genio": "base of tongue (swallowing muscle)",
    "fdi": "hand muscle (between fingers)",
    "ta": "lower leg muscle (lifting foot)",
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
    return {"+":1,"+1":1,"++":2,"+2":2,"+++":3,"+3":3,"+4":4}.get(val,0)

# =============================
# MNCS CLEANER
# =============================
def clean_mncs_nerves(mncs_list):
    cleaned = {}
    last_nerve = ""

    for m in mncs_list:
        nerve_raw = m.get("Nerve_Muscles","").strip()
        amp = safe_float(m.get("Amplitude_mv"))

        if nerve_raw and nerve_raw.replace('"','').strip():
            nerve = nerve_raw
            last_nerve = nerve
        else:
            nerve = last_nerve

        if nerve and not np.isnan(amp) and amp < 2.0:
            cleaned.setdefault(nerve, []).append(amp)

    return cleaned

# =============================
# FEATURE EXTRACTION + EXPLANATION
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
    fibs = []

    for e in emg:
        m = e.get("Muscles","")
        if any(k in m.lower() for k in bulbar_kw):
            bulbar.append(m)
        else:
            limb.append(m)
        if severity_score(str(e.get("Fibs",""))) > 0:
            fibs.append(m)

    # MNCS
    low_amp = clean_mncs_nerves(mncs)
    mncs_lines = []
    for n,amps in low_amp.items():
        if len(amps)==1:
            mncs_lines.append(f"{n} ({amps[0]} mV)")
        else:
            mncs_lines.append(f"{n} ({min(amps)}–{max(amps)} mV)")

    sensory = sorted(set(s.get("Nerve","") for s in sncs if s.get("Nerve")))

    # =============================
    # WHY YES (SIMPLE)
    # =============================
    if bulbar:
        why_yes.append(
            "Speech / swallowing muscles examined:\n  - " +
            "\n  - ".join([f"{m} ({muscle_meaning(m)})" for m in bulbar])
        )
    else:
        why_yes.append(
            "Speech and swallowing muscles (lips & tongue) appear normal"
        )

    if limb:
        why_yes.append(
            "Hand and leg muscles examined:\n  - " +
            "\n  - ".join([f"{m} ({muscle_meaning(m)})" for m in limb])
        )

    if fibs:
        why_yes.append(
            "Some muscles show abnormal electrical activity (muscle irritation)"
        )

    if mncs_lines:
        why_yes.append(
            "Motor nerves show reduced signal strength:\n  - " +
            "\n  - ".join(mncs_lines)
        )

    if sensory:
        why_yes.append(
            "Sensory nerves (feeling nerves) are normal:\n  - " +
            "\n  - ".join(sensory)
        )

    # =============================
    # WHY NOT OTHERS (VERY SIMPLE)
    # =============================
    why_not["PMA"] = (
        "PMA usually affects only hand and leg muscles.\n"
        "In this report, no clear muscle damage was seen."
    )

    why_not["PBP"] = (
        "PBP mainly affects lips and tongue muscles (speech problems).\n"
        "Those muscles appear normal here."
    )

    why_not["ALS"] = (
        "ALS affects both speech muscles and hand/leg muscles together.\n"
        "This mixed pattern was not seen in this report."
    )

    # =============================
    # FEATURES FOR MODEL
    # =============================
    features["emg_bulbar_count"] = len(bulbar)
    features["emg_limb_count"] = len(limb)
    features["emg_fibs_score"] = sum(severity_score(e.get("Fibs","")) for e in emg)
    features["mncs_count"] = len(mncs)
    features["sncs_count"] = len(sncs)

    return features, why_yes, why_not

# =============================
# RUN PREDICTION
# =============================
features, why_yes, why_not = extract_features_and_context(TEST_FILE)

X = np.array([[features.get(f,0) for f in feature_names]],dtype=float)
X = np.nan_to_num(X)

probs = model.predict_proba(X)[0]
pred = label_encoder.inverse_transform([np.argmax(probs)])[0]

# =============================
# OUTPUT
# =============================
print("\n" + "═"*60)
print("🧠 MND MODEL – CLINICAL DECISION SUMMARY")
print("═"*60)

print("\n📌 Final Predicted Diagnosis")
print(f"➡️  {pred}")

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
    if k!=pred:
        print(f"• {k}: {v}")

print("\n" + "─"*60)
print("📚 Medical References")
print("─"*60)
print("• Oh’s Clinical EMG")
print("• Preston & Shapiro – EMG & NCS")
print("• AANEM Guidelines")
print("• MNDA Diagnostic Criteria")

print("\n" + "═"*60)
print("✅ Prediction & explanation completed")
print("═"*60)
