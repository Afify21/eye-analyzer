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
REDNESS_THRESH = 0.15  # 2% Redness (Strict for Sclera Lock)
FATIGUE_THRESH = 130   # Brightness < 100 means Fatigue

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
        
        # 6. DUAL DIAGNOSIS (The Update)
        is_fatigued = brightness < FATIGUE_THRESH
        is_irritated = red_ratio > REDNESS_THRESH
        
        if is_fatigued and is_irritated:
            status = "Fatigue & Irritated"
        elif is_fatigued:
            status = "Fatigue"
        elif is_irritated:
            status = "Irritated"
        else:
            status = "Normal"
            
        return {
            "status": status,
            "redness": red_ratio,
            "brightness": brightness,
            "crop": crop,
            "vessel_mask": true_vessels
        }

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
            
            # --- COLOR LOGIC FOR DUAL STATE ---
            color = (0, 255, 0) # Green (Normal)
            
            if "Fatigue" in data['status'] and "Irritated" in data['status']:
                color = (255, 165, 0) # Orange (Both)
            elif "Irritated" in data['status']:
                color = (255, 0, 0)   # Red
            elif "Fatigue" in data['status']:
                color = (255, 255, 0) # Yellow
            
            cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), color, 2)
            
            # Use smaller font if text is long
            font_scale = 0.4 if "&" in data['status'] else 0.6
            cv2.putText(frame_rgb, f"{data['status']}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

        return frame_rgb, results

# --- VISUALIZATION ---
def show_full_report(orig_img, eye_results, filename):
    num_eyes = len(eye_results)
    if num_eyes == 0:
        print(f"❌ No eyes found in {filename}")
        return

    total_rows = 1 + num_eyes
    fig = plt.figure(figsize=(10, 4 * total_rows))
    
    # 1. Main Image
    ax_main = plt.subplot2grid((total_rows, 3), (0, 0), colspan=3)
    ax_main.imshow(orig_img)
    ax_main.set_title(f"Main Detection: {filename}")
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
        
        # Stats
        ax_text = plt.subplot2grid((total_rows, 3), (row, 2))
        ax_text.text(0.1, 0.5, 
                     f"DIAGNOSIS: {res['status']}\n\n"
                     f"Redness: {res['redness']*100:.2f}%\n"
                     f"Brightness: {res['brightness']:.0f}\n",
                     fontsize=12, va='center')
        ax_text.axis('off')

    plt.tight_layout()
    plt.show()

# --- MAIN ---
if __name__ == "__main__":
    lab = EyeLab()
    
    # YOUR FILES
    my_files = ['normal.jpg', 'red2.jpg', 'irritated2.jpg'] 
    
    for f in my_files:
        print(f"\n--- Processing {f} ---")
        img_res, results = lab.process_image(f)
        
        if img_res is not None:
            for res in results:
                print(f"  👁️ Eye {res['id']}: [{res['status']}] (R: {res['redness']*100:.1f}%)")
            
            show_full_report(img_res, results, f)