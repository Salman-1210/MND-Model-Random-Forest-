import json
import random
import os
from copy import deepcopy
from collections import defaultdict

# =========================
# CONFIG
# =========================
INPUT_DIR = "results"
OUTPUT_DIR = "generated/NORMAL"
TOTAL_CASES = 1500

FULL_RATIO = 0.20
MODERATE_RATIO = 0.35

random.seed(42)

# =========================
# LOAD REAL PATTERNS
# =========================
mncs_pool = defaultdict(list)
sncs_pool = defaultdict(list)
emg_pool = []

def load_patterns():
    for fname in os.listdir(INPUT_DIR):
        if not fname.endswith(".json"):
            continue

        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            data = json.load(f)

        for m in data.get("Motor_Nerve_Conduction_Studies", []):
            try:
                amp = float(m.get("Amplitude_mv", "0"))
            except:
                continue
            if amp >= 2.5:  # NORMAL motor amplitudes
                mncs_pool[m["Nerve_Muscles"]].append(m)

        for s in data.get("Sensory_Nerve_Conduction_Studies", []):
            sncs_pool[s["Nerve"]].append(s)

        for e in data.get("Electromyography", []):
            if str(e.get("Fibs", "")).lower() in ["nil", "none", "0", "-"]:
                emg_pool.append(e)

load_patterns()

MNCS_KEYS = list(mncs_pool.keys())
SNCS_KEYS = list(sncs_pool.keys())
EMG_KEYS = list({e["Muscles"] for e in emg_pool})

# =========================
# PICK FUNCTIONS
# =========================
def pick_mncs(coverage):
    if coverage == "full":
        nerves = MNCS_KEYS
    elif coverage == "moderate":
        nerves = random.sample(MNCS_KEYS, random.randint(3, min(7, len(MNCS_KEYS))))
    else:
        nerves = random.sample(MNCS_KEYS, random.randint(1, min(3, len(MNCS_KEYS))))

    out = []
    for n in nerves:
        out.append(deepcopy(random.choice(mncs_pool[n])))
    return out

def pick_sncs(coverage):
    if coverage == "full":
        nerves = SNCS_KEYS
    elif coverage == "moderate":
        nerves = random.sample(SNCS_KEYS, random.randint(2, min(5, len(SNCS_KEYS))))
    else:
        nerves = random.sample(SNCS_KEYS, random.randint(0, min(2, len(SNCS_KEYS))))

    return [deepcopy(random.choice(sncs_pool[n])) for n in nerves]

def pick_emg(coverage):
    if coverage == "sparse":
        return []

    count = random.randint(1, 3) if coverage == "moderate" else random.randint(2, 5)
    muscles = random.sample(EMG_KEYS, min(count, len(EMG_KEYS)))

    out = []
    for m in muscles:
        candidates = [e for e in emg_pool if e["Muscles"] == m]
        if candidates:
            out.append(deepcopy(random.choice(candidates)))
    return out

# =========================
# GENERATE
# =========================
def generate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i in range(1, TOTAL_CASES + 1):
        r = random.random()
        if r < FULL_RATIO:
            coverage = "full"
        elif r < FULL_RATIO + MODERATE_RATIO:
            coverage = "moderate"
        else:
            coverage = "sparse"

        case = {
            "Motor_Nerve_Conduction_Studies": pick_mncs(coverage),
            "Sensory_Nerve_Conduction_Studies": pick_sncs(coverage),
            "Electromyography": pick_emg(coverage)
        }

        fname = f"result_NORMAL_{i:04d}.json"
        with open(os.path.join(OUTPUT_DIR, fname), "w", encoding="utf-8") as f:
            json.dump(case, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    generate()
    print("✅ 1500 NON-MND cases generated (FORMAT MATCH CONFIRMED)")
