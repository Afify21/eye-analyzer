import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- PATH SETUP ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
except NameError:
    print("⚠️ Jupyter mode detected.")

# --- CONFIGURATION ---
# Severity Thresholds for Redness
REDNESS_MILD = 0.10      # 15% - Mild irritation
REDNESS_MODERATE = 0.15  # 18% - Moderate irritation
REDNESS_SEVERE = 0.19    # 36% - Severe irritation

# Severity Thresholds for Fatigue (Brightness)
FATIGUE_MILD = 130       # Brightness < 130 = Mild fatigue
FATIGUE_MODERATE = 110   # Brightness < 110 = Moderate fatigue
FATIGUE_SEVERE = 90      # Brightness < 90 = Severe fatigue

SYMMETRY_TOLERANCE = 0.02 # 2% difference triggers alert

class EyeLab:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def analyze_single_eye(self, eye_roi):
        """
        Runs the SCLERA-LOCKED pipeline on a SINGLE eye crop.
        """
        # 1. CROP: Aggressive crop (Top/Bottom 25%)
        h, w, _ = eye_roi.shape
        y1, y2 = int(h*0.25), int(h*0.75) 
        crop = eye_roi[y1:y2, :]
        
        # Convert to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # --- 2. SCLERA (WHITE) DETECTION ---
        lower_white = np.array([0, 0, 120]) 
        upper_white = np.array([180, 60, 255])
        sclera_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        kernel = np.ones((3,3), np.uint8)
        sclera_context = cv2.dilate(sclera_mask, kernel, iterations=2)

        # --- 3. RED VESSEL DETECTION ---
        lower_red1 = np.array([0, 70, 100]); upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 100]); upper_red2 = np.array([180, 255, 255])
        
        raw_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        
        # --- 4. THE INTERSECTION ---
        true_vessels = cv2.bitwise_and(raw_red, raw_red, mask=sclera_context)
        
        # 5. CALCULATE STATS
        red_pixels = cv2.countNonZero(true_vessels)
        sclera_area = cv2.countNonZero(sclera_context)
        
        if sclera_area == 0: sclera_area = 1 
        
        red_ratio = red_pixels / sclera_area
        
        # Brightness (Average of the white area only)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        brightness = cv2.mean(gray, mask=sclera_context)[0]
        
        # 6. SEVERITY CLASSIFICATION
        # Determine Irritation Severity
        if red_ratio >= REDNESS_SEVERE:
            irritation_level = "Severe"
        elif red_ratio >= REDNESS_MODERATE:
            irritation_level = "Moderate"
        elif red_ratio >= REDNESS_MILD:
            irritation_level = "Mild"
        else:
            irritation_level = None
        
        # Determine Fatigue Severity
        if brightness < FATIGUE_SEVERE:
            fatigue_level = "Severe"
        elif brightness < FATIGUE_MODERATE:
            fatigue_level = "Moderate"
        elif brightness < FATIGUE_MILD:
            fatigue_level = "Mild"
        else:
            fatigue_level = None
        
        # Combined Status with Severity
        if fatigue_level and irritation_level:
            status = f"{fatigue_level} Fatigue & {irritation_level} Irritation"
            severity = max(self._severity_score(fatigue_level), self._severity_score(irritation_level))
        elif fatigue_level:
            status = f"{fatigue_level} Fatigue"
            severity = self._severity_score(fatigue_level)
        elif irritation_level:
            status = f"{irritation_level} Irritation"
            severity = self._severity_score(irritation_level)
        else:
            status = "Normal"
            severity = 0
            
        return {
            "status": status,
            "severity": severity,
            "irritation_level": irritation_level,
            "fatigue_level": fatigue_level,
            "redness": red_ratio,
            "brightness": brightness,
            "crop": crop,
            "vessel_mask": true_vessels
        }
    
    def _severity_score(self, level):
        """Convert severity level to numeric score for comparison"""
        scores = {"Mild": 1, "Moderate": 2, "Severe": 3}
        return scores.get(level, 0)
    
    def get_recommendations(self, results):
        """Generate personalized recommendations based on eye analysis"""
        recommendations = []
        max_severity = 0
        has_fatigue = False
        has_irritation = False
        asymmetric = False
        
        # Analyze all eyes
        for res in results:
            if res['severity'] > max_severity:
                max_severity = res['severity']
            if res['fatigue_level']:
                has_fatigue = True
            if res['irritation_level']:
                has_irritation = True
        
        # Check asymmetry
        if len(results) == 2:
            diff = abs(results[0]['redness'] - results[1]['redness'])
            if diff > SYMMETRY_TOLERANCE:
                asymmetric = True
        
        # Generate recommendations based on conditions
        if max_severity >= 3:  # Severe
            recommendations.append("URGENT: Severe eye condition detected")
            recommendations.append("Consult an eye doctor immediately")
            recommendations.append("Avoid rubbing your eyes")
        
        if has_fatigue and has_irritation:
            recommendations.append("Take an immediate 10-15 minute break")
            recommendations.append("Use lubricating eye drops")
            recommendations.append("Consider getting more sleep tonight")
        elif has_fatigue:
            if max_severity >= 2:  # Moderate or Severe
                recommendations.append("Take a 10-minute break from screens")
                recommendations.append("Rest in a dark, quiet room")
            else:
                recommendations.append("Take a 5-minute break")
                recommendations.append("Follow the 20-20-20 rule: Every 20 min, look 20 feet away for 20 sec")
        elif has_irritation:
            if max_severity >= 2:
                recommendations.append("Apply preservative-free eye drops")
                recommendations.append("Use a warm compress for 5-10 minutes")
            else:
                recommendations.append("Consider using lubricating eye drops")
            recommendations.append("Avoid smoke, dust, and allergens")
        
        if asymmetric:
            recommendations.append("Asymmetric redness detected - may indicate localized issue")
            if max_severity >= 2:
                recommendations.append("Monitor closely; see a doctor if it persists")
        
        # General wellness tips
        if has_fatigue or has_irritation:
            recommendations.append("Reduce screen brightness")
            recommendations.append("Ensure proper lighting (avoid glare)")
            recommendations.append("Stay hydrated - drink water")
        
        if not recommendations:
            recommendations.append("Eyes appear healthy!")
            recommendations.append("Keep maintaining good eye care habits")
        
        return recommendations

    def process_image(self, img_path):
        frame = cv2.imread(img_path)
        if frame is None: return None, []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- DETECTION PHASE ---
        detected_eyes = [] 
        
        # Plan A: Face -> Eyes
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        for (fx, fy, fw, fh) in faces:
            roi_gray = gray[fy:fy+fh, fx:fx+fw]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 6, minSize=(30, 30))
            for (ex, ey, ew, eh) in eyes:
                detected_eyes.append((fx+ex, fy+ey, ew, eh))
        
        # Plan B: Whole Image
        if not detected_eyes:
            detected_eyes_rects = self.eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
            for (ex, ey, ew, eh) in detected_eyes_rects:
                detected_eyes.append((ex, ey, ew, eh))

        # Fallback
        if not detected_eyes:
             h, w, _ = frame.shape
             detected_eyes.append((0, 0, w, h))

        # Sort Left to Right
        detected_eyes.sort(key=lambda b: b[0])
        
        # --- ANALYSIS PHASE ---
        results = []
        
        for i, (x, y, w, h) in enumerate(detected_eyes):
            eye_roi = frame[y:y+h, x:x+w]
            data = self.analyze_single_eye(eye_roi)
            data['box'] = (x, y, w, h)
            data['id'] = i + 1
            results.append(data)
            
            # --- COLOR LOGIC BASED ON SEVERITY ---
            color = (0, 255, 0) # Green (Normal)
            thickness = 2  # Default thickness
            
            # Check for Severe Irritation first (Red with thicker box)
            if "Severe" in data['status'] and "Irritation" in data['status']:
                color = (255, 0, 0)   # Red for Severe Irritation
                thickness = 4         # Thicker box for emphasis
            elif "Fatigue" in data['status'] and "Irritation" in data['status']:
                color = (255, 165, 0) # Orange (Both)
            elif "Irritation" in data['status']:
                color = (255, 0, 0)   # Red (Any Irritation)
            elif "Fatigue" in data['status']:
                color = (255, 255, 0) # Yellow (Fatigue only)
            
            cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), color, thickness)
            
            # Use smaller font if text is long
            font_scale = 0.4 if "&" in data['status'] else 0.6
            cv2.putText(frame_rgb, f"{data['status']}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

        # --- SYMMETRY CHECK ---
        if len(results) == 2:
            diff = abs(results[0]['redness'] - results[1]['redness'])
            if diff > SYMMETRY_TOLERANCE:
                h_img, w_img, _ = frame_rgb.shape
                cv2.putText(frame_rgb, "SYMMETRY ALERT!", (int(w_img/2)-100, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                print(f"  WARNING: Symmetry Alert: Redness Diff {diff*100:.1f}%")

        # --- GENERATE RECOMMENDATIONS ---
        recommendations = self.get_recommendations(results)

        return frame_rgb, results, recommendations

# --- VISUALIZATION ---
def show_full_report(orig_img, eye_results, filename, recommendations):
    num_eyes = len(eye_results)
    if num_eyes == 0:
        print(f"ERROR: No eyes found in {filename}")
        return

    total_rows = 1 + num_eyes + 1  # +1 for recommendations row
    fig = plt.figure(figsize=(10, 4 * total_rows))
    
    # 1. Main Image
    ax_main = plt.subplot2grid((total_rows, 3), (0, 0), colspan=3)
    ax_main.imshow(orig_img)
    ax_main.set_title(f"Main Detection: {filename}", fontsize=14, fontweight='bold')
    ax_main.axis('off')
    
    # 2. Individual Eyes
    for i, res in enumerate(eye_results):
        row = i + 1
        
        # Crop
        ax_crop = plt.subplot2grid((total_rows, 3), (row, 0))
        ax_crop.imshow(cv2.cvtColor(res['crop'], cv2.COLOR_BGR2RGB))
        ax_crop.set_title(f"Eye {res['id']} (Cropped)")
        ax_crop.axis('off')
        
        # Mask
        ax_mask = plt.subplot2grid((total_rows, 3), (row, 1))
        ax_mask.imshow(res['vessel_mask'], cmap='gray')
        ax_mask.set_title("Sclera-Locked Vessels")
        ax_mask.axis('off')
        
        # Stats with Severity
        ax_text = plt.subplot2grid((total_rows, 3), (row, 2))
        severity_labels = ["Normal", "Mild", "Moderate", "Severe"]
        
        stats_text = f"DIAGNOSIS: {res['status']}\n\n"
        stats_text += f"Redness: {res['redness']*100:.2f}%\n"
        stats_text += f"Brightness: {res['brightness']:.0f}\n"
        stats_text += f"Severity: {severity_labels[res['severity']]}\n"
        
        ax_text.text(0.1, 0.5, stats_text, fontsize=11, va='center', family='monospace')
        ax_text.axis('off')

    # 3. Recommendations Section
    rec_row = 1 + num_eyes
    ax_rec = plt.subplot2grid((total_rows, 3), (rec_row, 0), colspan=3)
    
    rec_text = "RECOMMENDATIONS:\n" + "="*50 + "\n\n"
    for rec in recommendations:
        rec_text += f"{rec}\n"
    
    ax_rec.text(0.05, 0.95, rec_text, fontsize=11, va='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax_rec.set_title("Action Plan", fontsize=13, fontweight='bold')
    ax_rec.axis('off')

    plt.tight_layout()
    plt.show()

# --- MAIN ---
if __name__ == "__main__":
    lab = EyeLab()
    
    # AUTO-DETECT ALL JPG FILES
    my_files = [f for f in os.listdir('.') if f.lower().endswith('.jpg')]
    my_files.sort()
    
    if not my_files:
        print("No .jpg files found in directory.")

    for f in my_files:
        print(f"\n{'='*60}")
        print(f"Processing {f}")
        print('='*60)
        img_res, results, recommendations = lab.process_image(f)
        
        if img_res is not None:
            # Print eye analysis
            for res in results:
                severity_labels = ["Normal", "Mild", "Moderate", "Severe"]
                print(f"  Eye {res['id']}: [{res['status']}] - Severity: {severity_labels[res['severity']]}")
                print(f"     Redness: {res['redness']*100:.1f}% | Brightness: {res['brightness']:.0f}")
            
            # Print recommendations
            print(f"\nRECOMMENDATIONS:")
            for rec in recommendations:
                print(f"  {rec}")
            
            show_full_report(img_res, results, f, recommendations)