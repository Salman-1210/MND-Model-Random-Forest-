"""
extract_table_medical_report.py

Usage:
- Put your image file (e.g. medicalreport.jpg) in the script folder.
- Adjust TESSERACT_CMD if needed.
- Run: python extract_table_medical_report.py
Outputs:
- ocr_output.txt      (raw OCR of each cell in table order)
- structured_output.json   (the JSON in the format you requested)
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract
import json
import os
import re

# === CONFIG ===
IMAGE_PATH = "medical_report.jpg"   # change to your image filename
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # change if needed
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Tesseract config for numbers + text (try to keep as general)
TESS_CONFIG = r'--psm 6'  # assumes a single uniform block of text in each crop

# === UTILITIES ===
def cleanup_text(s):
    if s is None:
        return ""
    # Remove weird control chars, collapse whitespace
    s = re.sub(r'[^\x00-\x7F]+', ' ', s)
    s = s.replace('\n', ' ').strip()
    s = re.sub(r'\s{2,}', ' ', s)
    return s

def sort_cells_by_row_col(cells):
    # cells: list of (x, y, w, h, text)
    # We'll group by y (rows) then sort by x inside each row
    # Use y center to group rows robustly
    rows = []
    for c in cells:
        x,y,w,h,t = c
        cy = y + h/2
        placed = False
        for row in rows:
            # if center y close to row y, append (tolerance)
            if abs(row['y'] - cy) < max(10, h*0.4):
                row['cells'].append(c)
                # update average y
                row['y'] = (row['y'] * (len(row['cells'])-1) + cy) / len(row['cells'])
                placed = True
                break
        if not placed:
            rows.append({'y':cy, 'cells':[c]})
    # sort rows by y
    rows = sorted(rows, key=lambda r: r['y'])
    # sort cells inside row by x
    sorted_grid = []
    for r in rows:
        row_cells = sorted(r['cells'], key=lambda c: c[0])  # sort by x
        sorted_grid.append(row_cells)
    return sorted_grid

# === MAIN TABLE DETECTION & OCR ===
def detect_table_and_read(image_path):
    # Read image
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold to get binary image
    # Use inverse binary because lines are dark on white
    thresh = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, -2)

    # Detect horizontal and vertical lines by morphological ops
    horizontal = thresh.copy()
    vertical = thresh.copy()

    cols = horizontal.shape[1]
    horizontal_size = max(10, cols // 30)
    horiz_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horiz_structure)
    horizontal = cv2.dilate(horizontal, horiz_structure)

    rows = vertical.shape[0]
    vertical_size = max(10, rows // 30)
    vert_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, vert_structure)
    vertical = cv2.dilate(vertical, vert_structure)

    # Combine detected lines
    mask = horizontal + vertical

    # Find contours of table-like cells
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size (ignore tiny blobs)
    boxes = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        if area < 1000:   # tune threshold if needed
            continue
        # exclude the whole page bounding box (very large)
        if w > orig.shape[1]*0.9 and h > orig.shape[0]*0.9:
            continue
        boxes.append((x,y,w,h))

    # If no boxes found, fallback: try connected components on threshold
    if not boxes:
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(~gray, connectivity=8)
        for i in range(1, nb_components):
            x,y,w,h,area = stats[i]
            if area < 1000: 
                continue
            boxes.append((x,y,w,h))

    # Merge overlapping boxes (cells from same row/col)
    # Expand slightly then group by overlap
    boxes_np = np.array(boxes, dtype=int)
    # Remove duplicates
    boxes_np = np.unique(boxes_np, axis=0).tolist()

    # Finally crop each box and OCR it
    cells = []
    for (x,y,w,h) in boxes_np:
        # enlarge box a bit for safety
        pad_w = int(w * 0.06)
        pad_h = int(h * 0.06)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(orig.shape[1], x + w + pad_w)
        y2 = min(orig.shape[0], y + h + pad_h)
        crop = orig[y1:y2, x1:x2]
        # convert to PIL for pytesseract
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)
        raw = pytesseract.image_to_string(pil_crop, config=TESS_CONFIG, lang='eng')
        text = cleanup_text(raw)
        cells.append((x1, y1, x2-x1, y2-y1, text))

    # Sort into grid
    grid = sort_cells_by_row_col(cells)
    return grid, orig

# === MAP GRID -> JSON FORMAT ===
def grid_to_structured_json(grid):
    """
    Expect first row = header in Motor table.
    We'll read each subsequent row and map columns.
    The Motor table header in your document uses columns:
    Nerve-Muscles | Stimulus site | Latency (ms) | Distance (cm) | Amplitude (mv) | NCV (ms) | F.LAT. (ms) | Duration (ms)
    We'll map first 6 columns to the Motor JSON you requested.
    For Sensory and EMG, similar approach will be used.
    """
    motor = []
    sensory = []
    emg = []

    # Heuristic: find the motor header row by checking for "Nerve" or "Nerve-Muscles" in any cell
    header_row_idx = None
    for i, row in enumerate(grid):
        concat = " ".join([c[4].lower() for c in row])
        if "nerve" in concat and "latency" in concat:
            header_row_idx = i
            break
    if header_row_idx is None:
        # fallback - assume top part is motor table header
        header_row_idx = 0

    # Find approximate row ranges: motor header -> next rows until a blank row -> sensory header -> emg header
    # Build a list of strings per row for easier matching
    row_texts = [" ".join([c[4] for c in r]).strip() for r in grid]

    # Determine motor rows: rows after header until we hit "sensory"
    motor_rows = []
    sensory_rows = []
    emg_rows = []

    # locate sensory and emg indicators
    sensory_idx = None
    emg_idx = None
    for idx, t in enumerate(row_texts):
        low = t.lower()
        if "sensory nerve conduction" in low or "sensory" in low and "latency" in low:
            sensory_idx = idx
            break
    for idx, t in enumerate(row_texts):
        if "electromyograph" in t.lower() or "electromyography" in t.lower() or "spontaneous" in t.lower():
            emg_idx = idx
            break

    # motor between header_row_idx and sensory_idx (or emg_idx or end)
    end_motor = sensory_idx if sensory_idx is not None else (emg_idx if emg_idx is not None else len(grid))
    for r in grid[header_row_idx+1:end_motor]:
        motor_rows.append(r)

    # sensory between sensory_idx and emg_idx
    if sensory_idx is not None:
        end_sensory = emg_idx if emg_idx is not None else len(grid)
        for r in grid[sensory_idx+1:end_sensory]:
            sensory_rows.append(r)

    # emg from emg_idx onwards
    if emg_idx is not None:
        for r in grid[emg_idx+1:]:
            emg_rows.append(r)

    # Helper to extract cell by column index (safe)
    def text_at(row, idx):
        if idx < len(row):
            return row[idx][4].strip()
        return ""

    # Parse motor rows - we assume first 6 columns as requested (if there are extra columns, ignore)
    for row in motor_rows:
        # remove empty rows or noise
        joined = " ".join([c[4] for c in row]).strip()
        if len(joined) < 2:
            continue
        # Try to locate column boundaries: look for the cell that contains "wrist", "elbow", "ankle", etc. to guess Stimulus_Site
        # We'll map by columns: first cell = nerve, second cell = site, third = latency, fourth = distance, fifth = amplitude, sixth = NCV
        nerve = text_at(row, 0)
        site = text_at(row, 1)
        latency = text_at(row, 2)
        distance = text_at(row, 3)
        amplitude = text_at(row, 4)
        ncv = text_at(row, 5)

        # If nerve cell is empty but first cell seems like "Rt. Median APB" joined across two cells, join adjacent cells
        if not nerve and len(row) >= 2:
            nerve = (text_at(row,0) + " " + text_at(row,1)).strip()

        # Basic normalization: replace empty with NR
        def norm(v): 
            v = v.strip()
            return v if v else "NR"

        motor.append({
            "Nerve_Muscles": norm(nerve),
            "Stimulus_Site": norm(site),
            "Latency_ms": norm(latency),
            "Distance_cm": norm(distance),
            "Amplitude_mv": norm(amplitude),
            "NCV_ms": norm(ncv)
        })

    # Parse sensory rows - expect columns: Nerve | Recording Site | Stimulation Site | Latency | Distance | Amplitude | NCV
    for row in sensory_rows:
        joined = " ".join([c[4] for c in row]).strip()
        if len(joined) < 2:
            continue
        nerve = text_at(row,0)
        rec_site = text_at(row,1)
        stim_site = text_at(row,2)
        latency = text_at(row,3)
        distance = text_at(row,4)
        amplitude = text_at(row,5) if len(row) > 5 else ""
        ncv = text_at(row,6) if len(row) > 6 else ""

        def norm(v): 
            v = v.strip()
            return v if v else "NR"

        sensory.append({
            "Nerve": norm(nerve),
            "Recording_Site": norm(rec_site),
            "Stimulation_Site": norm(stim_site),
            "Latency_ms": norm(latency),
            "Distance_cm": norm(distance),
            "Amplitude_uv": norm(amplitude),
            "NCV_ms": norm(ncv)
        })

    # Parse EMG rows - header likely contains 'Muscles' and columns for Fibs, Psw, Others, Amp, Duration, Polys, Recruit, Interference
    for row in emg_rows:
        joined = " ".join([c[4] for c in row]).strip()
        if len(joined) < 2:
            continue
        muscles = text_at(row,0)
        fibs = text_at(row,1)
        psw = text_at(row,2)
        others = text_at(row,3)
        amp = text_at(row,4)
        duration = text_at(row,5)
        polys = text_at(row,6) if len(row) > 6 else ""
        recruit = text_at(row,7) if len(row) > 7 else ""
        interference = text_at(row,8) if len(row) > 8 else ""

        def norm(v): 
            v = v.strip()
            return v if v else "NR"

        emg.append({
            "Muscles": norm(muscles),
            "Fibs": norm(fibs),
            "Psw": norm(psw),
            "Others": norm(others),
            "Amp": norm(amp),
            "Duration": norm(duration),
            "Polys": norm(polys),
            "Recruit": norm(recruit),
            "Interference": norm(interference)
        })

    return {
        "Motor_Nerve_Conduction_Studies": motor,
        "Sensory_Nerve_Conduction_Studies": sensory,
        "Electromyography": emg
    }

# === RUN ===
def main():
    print("Starting table detection + OCR...")
    grid, orig = detect_table_and_read(IMAGE_PATH)

    # Save raw OCR per cell for debugging
    with open("ocr_output.txt", "w", encoding="utf-8") as f:
        for r_idx, row in enumerate(grid):
            for c_idx, cell in enumerate(row):
                f.write(f"R{r_idx:02d}C{c_idx:02d}: {cell[4]}\n")
            f.write("\n")

    structured = grid_to_structured_json(grid)

    with open("structured_output.json", "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=4)

    print("Done. Saved 'ocr_output.txt' and 'structured_output.json'")

if __name__ == "__main__":
    main()
