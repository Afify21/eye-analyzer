import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd

# --- SAFE IMPORT FOR DLIB ---
try:
    import dlib
    from imutils import face_utils
except ImportError:
    print("\nCRITICAL ERROR: Dlib is not installed!")
    print("Please run: pip install dlib imutils opencv-python")
    input("Press Enter to exit...")
    sys.exit()

# --- CONFIGURATION ---
REDNESS_MILD = 0.30      # Changed from 0.05 based on data analysis
REDNESS_MODERATE = 0.50  # Changed from 0.12
REDNESS_SEVERE = 0.70    # Changed from 0.20

FATIGUE_MILD = 130
FATIGUE_MODERATE = 110
FATIGUE_SEVERE = 90


class EyeLab:
    def __init__(self):
        dat_file = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(dat_file):
            print(f"\n❌ ERROR: '{dat_file}' is missing!")
            raise FileNotFoundError("Missing landmark file")

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(dat_file)
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

    def get_dynamic_skin_mask(self, full_frame, eye_box):
        x, y, w, h = eye_box
        cheek_y = y + h
        cheek_h = int(h * 0.3)
        cheek_x = x + int(w * 0.2)
        cheek_w = int(w * 0.6)

        H, W = full_frame.shape[:2]
        if cheek_y + cheek_h > H:
            return None

        cheek_roi = full_frame[
            cheek_y:cheek_y + cheek_h,
            cheek_x:cheek_x + cheek_w
        ]

        hsv_cheek = cv2.cvtColor(cheek_roi, cv2.COLOR_BGR2HSV)
        mean, std = cv2.meanStdDev(hsv_cheek)
        mean = mean.flatten()
        std = std.flatten()

        lower_skin = mean - (2.5 * std)
        upper_skin = mean + (2.5 * std)

        lower_skin = np.clip(lower_skin, 0, [180, 255, 255])
        upper_skin = np.clip(upper_skin, 0, [180, 255, 255])

        return (
            np.array(lower_skin, dtype=np.uint8),
            np.array(upper_skin, dtype=np.uint8)
        )

    def analyze_single_eye(self, eye_roi, skin_ranges):
        # Resize eye ROI to be larger for better detection
        h, w = eye_roi.shape[:2]
        if h < 50 or w < 50:
            scale = max(50/h, 50/w)
            new_h, new_w = int(h*scale), int(w*scale)
            eye_roi = cv2.resize(eye_roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        hsv = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2HSV)

        # More aggressive sclera detection
        lower_body = np.array([0, 0, 80])      # Increased from 50
        upper_body = np.array([180, 80, 255])  # Decreased from 110
        sclera_body = cv2.inRange(hsv, lower_body, upper_body)

        if skin_ranges is not None:
            lower_skin, upper_skin = skin_ranges
            mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)
            kernel = np.ones((3, 3), np.uint8)
            mask_skin = cv2.dilate(mask_skin, kernel, iterations=2)
            sclera_body = cv2.subtract(sclera_body, mask_skin)

        h_roi, w_roi = sclera_body.shape
        # Less aggressive cropping
        sclera_body[:int(h_roi * 0.15), :] = 0  # Changed from 0.25
        sclera_body[int(h_roi * 0.85):, :] = 0  # Changed from 0.75

        cnts, _ = cv2.findContours(
            sclera_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if cnts:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            clean = np.zeros_like(sclera_body)
            for i in range(min(3, len(cnts))):  # Changed from 2 to 3
                cv2.drawContours(clean, cnts, i, 255, -1)
            sclera_body = clean

        # Larger ellipse mask
        ellipse = np.zeros_like(sclera_body)
        cv2.ellipse(
            ellipse,
            (w_roi // 2, h_roi // 2),
            (int(w_roi * 0.45), int(h_roi * 0.35)),  # Slightly smaller to focus on sclera
            0, 0, 360, 255, -1
        )
        sclera_body = cv2.bitwise_and(sclera_body, ellipse)

        # More lenient red detection
        lower_red1 = np.array([0, 30, 30])      # Lowered from [0, 50, 50]
        upper_red1 = np.array([10, 255, 255])   # Lowered from 15
        lower_red2 = np.array([170, 30, 30])    # Changed from [165, 50, 50]
        upper_red2 = np.array([180, 255, 255])

        red_mask = (
            cv2.inRange(hsv, lower_red1, upper_red1) +
            cv2.inRange(hsv, lower_red2, upper_red2)
        )
        true_redness = cv2.bitwise_and(red_mask, red_mask, mask=sclera_body)

        body_area = cv2.countNonZero(sclera_body) or 1
        red_area = cv2.countNonZero(true_redness)
        red_ratio = red_area / body_area

        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        brightness = cv2.mean(gray, mask=sclera_body)[0]

        irritation = None
        if red_ratio >= REDNESS_SEVERE: irritation = "Severe"
        elif red_ratio >= REDNESS_MODERATE: irritation = "Moderate"
        elif red_ratio >= REDNESS_MILD: irritation = "Mild"

        fatigue = None
        if brightness < FATIGUE_SEVERE: fatigue = "Severe"
        elif brightness < FATIGUE_MODERATE: fatigue = "Moderate"
        elif brightness < FATIGUE_MILD: fatigue = "Mild"

        if fatigue and irritation:
            status = f"{fatigue} Fatigue & {irritation} Irritation"
        elif fatigue:
            status = f"{fatigue} Fatigue"
        elif irritation:
            status = f"{irritation} Irritation"
        else:
            status = "Normal"

        severity = {"Mild": 1, "Moderate": 2, "Severe": 3}.get(irritation, 0)

        return {
            "status": status,
            "severity": severity,
            "redness": red_ratio,
            "brightness": brightness,
            "crop": eye_roi,
            "vessel_mask": true_redness,
            "sclera_mask": sclera_body
        }

    def get_recommendations(self, results):
        max_sev = max([r["severity"] for r in results], default=0)
        if max_sev >= 3: return ["URGENT: Severe inflammation detected."]
        if max_sev == 2: return ["Moderate: Take a break and use drops."]
        return ["Healthy: Keep it up!"]

    def process_image(self, img_path):
        frame = cv2.imread(img_path)
        if frame is None:
            return None, [], []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = []

        rects = self.detector(gray, 0)

        if rects:
            for rect in rects:
                shape = face_utils.shape_to_np(
                    self.predictor(gray, rect)
                )
                eyes = [shape[36:42], shape[42:48]]
                for pts in eyes:
                    x, y, w, h = cv2.boundingRect(pts)
                    eye_roi = frame[y:y + h, x:x + w]
                    if eye_roi.size == 0:
                        continue
                    skin = self.get_dynamic_skin_mask(frame, (x, y, w, h))
                    data = self.analyze_single_eye(eye_roi, skin)
                    data["box"] = (x, y, w, h)
                    results.append(data)

        rec = self.get_recommendations(results)
        return frame_rgb, results, rec


def predict_label_from_result(res):
    return "Irritated Only" if res["redness"] >= REDNESS_MILD else "Healthy"


def save_eye_outputs(filename, eye_idx, res, out_dir="dataset_outputs"):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(filename)[0]

    cv2.imwrite(
        os.path.join(out_dir, f"{base}_eye{eye_idx}_crop.png"),
        res["crop"]
    )
    cv2.imwrite(
        os.path.join(out_dir, f"{base}_eye{eye_idx}_sclera.png"),
        res["sclera_mask"]
    )
    cv2.imwrite(
        os.path.join(out_dir, f"{base}_eye{eye_idx}_vessels.png"),
        res["vessel_mask"]
    )


if __name__ == "__main__":
    lab = EyeLab()

    # ===== NORMAL MODE (UNCHANGED) =====
    files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.png'))]
    for f in files:
        img, res, rec = lab.process_image(f)
        if img is not None and res:
            pass

    # ===== DATASET MODE =====
    IMAGE_DIR = "Samples"
    CSV_PATH = "medical_eye_report (1).csv"

    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: CSV file '{CSV_PATH}' not found!")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"\n📊 Loaded CSV with {len(df)} rows")
    
    # Check if Samples directory exists
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ Error: Directory '{IMAGE_DIR}' not found!")
        sys.exit(1)
    
    # Build a mapping: folder_name -> {eye_1: path, eye_2: path, full_face: path}
    print(f"🔍 Scanning for images in '{IMAGE_DIR}' and subdirectories...")
    folder_map = {}
    subdirs_found = []
    subdirs_without_images = []
    
    # First, get all subdirectories
    all_subdirs = [d for d in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, d))]
    print(f"📂 Total subdirectories found: {len(all_subdirs)}")
    
    for subdir in all_subdirs:
        subdir_path = os.path.join(IMAGE_DIR, subdir)
        files = os.listdir(subdir_path)
        
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            subdirs_without_images.append(subdir)
            continue
        
        subdirs_found.append(subdir)
        folder_map[subdir] = {}
        
        for file in image_files:
            full_path = os.path.join(subdir_path, file)
            # Store by file type
            if 'eye_1' in file.lower():
                folder_map[subdir]['eye_1'] = full_path
            elif 'eye_2' in file.lower():
                folder_map[subdir]['eye_2'] = full_path
            elif 'full_face' in file.lower() or 'face' in file.lower():
                folder_map[subdir]['full_face'] = full_path
    
    print(f"✅ Found {len(folder_map)} subdirectories WITH images")
    if subdirs_without_images:
        print(f"⚠️  Found {len(subdirs_without_images)} subdirectories WITHOUT images")
        print(f"   First 10: {subdirs_without_images[:10]}")
    print(f"   First 10 with images: {subdirs_found[:10]}")
    
    # Try to understand the CSV filename pattern
    print("\n🔍 Analyzing CSV filename pattern:")
    print("   First 5 CSV filenames:")
    for filename in df["Filename"].head(5):
        base = os.path.splitext(filename)[0]  # Remove extension
        padded = base.zfill(5)  # Pad with zeros to 5 digits
        print(f"   - {filename} -> base: {base} -> padded: {padded}")
        if padded in folder_map:
            print(f"     ✓ MATCH! Found folder '{padded}'")
        else:
            print(f"     ✗ No folder '{padded}' found")
    
    # Check if any CSV filenames (without extension, zero-padded) match folder names
    print(f"\n   Available folders: {list(folder_map.keys())[:20]}")
    
    total = 0
    correct = 0
    errors = []
    processed = 0
    skipped = 0
    
    # Track redness distributions
    redness_healthy = []
    redness_irritated = []
    debug_samples = []  # Store first 5 for debugging

    print("\n⏳ Processing images...")
    for idx, row in df.iterrows():
        filename = row["Filename"]
        
        # Try to map CSV filename to folder
        # Remove extension and try zero-padding
        base = os.path.splitext(filename)[0]
        folder_candidates = [
            base,           # Try as-is
            base.zfill(5),  # Try with 5 digits
            base.zfill(6),  # Try with 6 digits
        ]
        
        folder_found = None
        for candidate in folder_candidates:
            if candidate in folder_map:
                folder_found = candidate
                break
        
        if not folder_found:
            skipped += 1
            continue
        
        # Use the full_face image for analysis
        if 'full_face' not in folder_map[folder_found]:
            skipped += 1
            continue
            
        path = folder_map[folder_found]['full_face']
        
        img, results, _ = lab.process_image(path)
        
        if not results:
            skipped += 1
            continue
        
        processed += 1

        for i, res in enumerate(results):
            eye_id = i + 1
            col_name = f"eye_{eye_id}_Final_Status"
            
            # Check if column exists and has value
            if col_name not in row.index or pd.isna(row[col_name]):
                continue
            
            save_eye_outputs(filename, eye_id, res)

            pred = predict_label_from_result(res)
            gt = row[col_name]
            
            # Track redness by ground truth
            if gt == "Healthy":
                redness_healthy.append(res['redness'])
            elif gt == "Irritated Only":
                redness_irritated.append(res['redness'])
            
            # Save debug info for first few samples
            if len(debug_samples) < 10:
                sclera_area = cv2.countNonZero(res['sclera_mask'])
                vessel_area = cv2.countNonZero(res['vessel_mask'])
                debug_samples.append({
                    'file': filename,
                    'eye': eye_id,
                    'gt': gt,
                    'redness': res['redness'],
                    'sclera_area': sclera_area,
                    'vessel_area': vessel_area
                })

            total += 1
            if pred == gt:
                correct += 1
            else:
                errors.append([filename, f"eye_{eye_id}", gt, pred, f"{res['redness']:.4f}"])
        
        # Progress indicator every 100 images
        if processed % 100 == 0:
            print(f"   Processed {processed} images...")

    acc = correct / total if total else 0
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Images processed: {processed}")
    print(f"Images skipped: {skipped}")
    print(f"Total eyes evaluated: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("="*60)
    
    # Show debug samples
    if debug_samples:
        print("\n" + "="*60)
        print("DEBUG: First 10 Samples")
        print("="*60)
        for s in debug_samples:
            print(f"{s['file']} eye_{s['eye']} ({s['gt']}):")
            print(f"  Sclera area: {s['sclera_area']} pixels")
            print(f"  Vessel area: {s['vessel_area']} pixels")
            print(f"  Redness ratio: {s['redness']:.4f} ({s['redness']*100:.1f}%)")
            if s['sclera_area'] == 0:
                print(f"  ⚠️  WARNING: No sclera detected!")
            if s['redness'] > 0.5:
                print(f"  ⚠️  WARNING: Very high redness!")
            print()
    
    # Show redness distribution analysis
    if redness_healthy and redness_irritated:
        print("\n" + "="*60)
        print("REDNESS DISTRIBUTION ANALYSIS")
        print("="*60)
        print(f"\nHealthy Eyes (n={len(redness_healthy)}):")
        print(f"  Mean: {np.mean(redness_healthy):.4f}")
        print(f"  Median: {np.median(redness_healthy):.4f}")
        print(f"  Std: {np.std(redness_healthy):.4f}")
        print(f"  Range: [{np.min(redness_healthy):.4f}, {np.max(redness_healthy):.4f}]")
        
        print(f"\nIrritated Eyes (n={len(redness_irritated)}):")
        print(f"  Mean: {np.mean(redness_irritated):.4f}")
        print(f"  Median: {np.median(redness_irritated):.4f}")
        print(f"  Std: {np.std(redness_irritated):.4f}")
        print(f"  Range: [{np.min(redness_irritated):.4f}, {np.max(redness_irritated):.4f}]")
        
        # Suggest optimal threshold
        max_healthy = np.max(redness_healthy)
        min_irritated = np.min(redness_irritated)
        suggested = (max_healthy + min_irritated) / 2
        
        print(f"\n📊 SUGGESTED THRESHOLD: {suggested:.4f}")
        print(f"   Current threshold: {REDNESS_MILD:.4f}")
        print(f"   Max healthy redness: {max_healthy:.4f}")
        print(f"   Min irritated redness: {min_irritated:.4f}")
        print("="*60)

    if errors:
        error_df = pd.DataFrame(
            errors,
            columns=["Filename", "Eye", "Actual", "Predicted", "Redness"]
        )
        error_df.to_csv("dataset_errors.csv", index=False)
        print(f"\n💾 Saved {len(errors)} errors to 'dataset_errors.csv'")
        print("\nFirst 10 errors:")
        print(error_df.head(10).to_string(index=False))
    else:
        print("\n🎉 No errors found!")