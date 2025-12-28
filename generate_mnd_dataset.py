import json
import random
import os
from copy import deepcopy
from collections import defaultdict

# =========================
# CONFIG
# =========================
INPUT_DIR = "results"
OUTPUT_DIR = "generated"
CASES_PER_TYPE = 500

FULL_CASE_RATIO = 0.15   # 15% full-coverage
MODERATE_RATIO = 0.30    # 30% moderate
# rest = sparse

random.seed(42)

# =========================
# STEP 1: EXTRACT REAL PATTERNS
# =========================
mncs_pool = defaultdict(lambda: defaultdict(list))
sncs_pool = defaultdict(list)
emg_pool = []

def load_real_patterns():
    for fname in os.listdir(INPUT_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            data = json.load(f)

        for m in data.get("Motor_Nerve_Conduction_Studies", []):
            key = m["Nerve_Muscles"]
            mncs_pool[key][m["Stimulus_Site"]].append(m)

        for s in data.get("Sensory_Nerve_Conduction_Studies", []):
            sncs_pool[s["Nerve"]].append(s)

        for e in data.get("Electromyography", []):
            emg_pool.append(e)

load_real_patterns()

MNCS_KEYS = list(mncs_pool.keys())
SNCS_KEYS = list(sncs_pool.keys())
EMG_KEYS = list({e["Muscles"] for e in emg_pool})

# =========================
# STEP 2: DISEASE LOGIC
# =========================
def select_emg(case_type, coverage):
    muscles = []

    if case_type == "PBP":
        bulbar = [m for m in EMG_KEYS if any(x in m.lower() for x in ["oris", "tongue", "genio"])]
        limb = [m for m in EMG_KEYS if m not in bulbar]

        muscles.extend(random.sample(bulbar, min(len(bulbar), random.randint(2, 4))))
        if coverage == "full":
            muscles.extend(random.sample(limb, min(2, len(limb))))
        else:
            muscles.extend(random.sample(limb, min(1, len(limb))))

    elif case_type == "PMA":
        limb = [m for m in EMG_KEYS if not any(x in m.lower() for x in ["oris", "tongue", "genio"])]
        muscles.extend(random.sample(limb, random.randint(3, 6)))

    elif case_type == "ALS":
        muscles.extend(random.sample(EMG_KEYS, random.randint(4, min(8, len(EMG_KEYS)))))

    return muscles

def pick_emg_entries(muscles):
    entries = []
    for m in muscles:
        candidates = [e for e in emg_pool if e["Muscles"] == m]
        if candidates:
            entries.append(deepcopy(random.choice(candidates)))
    return entries

# =========================
# STEP 3: MNCS / SNCS
# =========================
def pick_mncs(coverage):
    keys = MNCS_KEYS if coverage == "full" else random.sample(MNCS_KEYS, random.randint(2, min(6, len(MNCS_KEYS))))
    entries = []
    for k in keys:
        sites = list(mncs_pool[k].values())
        chosen = random.choice(sites)
        entries.append(deepcopy(random.choice(chosen)))
    return entries

def pick_sncs(coverage):
    if coverage == "full":
        nerves = SNCS_KEYS
    else:
        nerves = random.sample(SNCS_KEYS, random.randint(0, min(4, len(SNCS_KEYS))))

    entries = []
    for n in nerves:
        entries.append(deepcopy(random.choice(sncs_pool[n])))
    return entries

# =========================
# STEP 4: GENERATE DATASET
# =========================
def generate_cases(case_type):
    out_dir = os.path.join(OUTPUT_DIR, case_type)
    os.makedirs(out_dir, exist_ok=True)

    for i in range(1, CASES_PER_TYPE + 1):
        r = random.random()
        if r < FULL_CASE_RATIO:
            coverage = "full"
        elif r < FULL_CASE_RATIO + MODERATE_RATIO:
            coverage = "moderate"
        else:
            coverage = "sparse"

        emg_muscles = select_emg(case_type, coverage)

        case = {
            "Motor_Nerve_Conduction_Studies": pick_mncs(coverage),
            "Sensory_Nerve_Conduction_Studies": pick_sncs(coverage),
            "Electromyography": pick_emg_entries(emg_muscles)
        }

        fname = f"result_{case_type}_{i:03d}.json"
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            json.dump(case, f, indent=2, ensure_ascii=False)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    for disease in ["ALS", "PMA", "PBP"]:
        generate_cases(disease)
    print("✅ Dataset generation complete.")
