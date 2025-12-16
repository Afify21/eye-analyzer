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
REDNESS_THRESH = 0.2  # 3% redness (Strict because we remove skin)
FATIGUE_THRESH = 80   # Brightness < 100 means closed/dark

class EyeLab:
    def __init__(self):
        # Load Haar Cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def analyze_single_eye(self, eye_roi):
        """
        Runs the full medical pipeline on a SINGLE eye crop.
        """
        # 1. CROP: Remove Eyelids (Top 20%, Bottom 20%)
        h, w, _ = eye_roi.shape
        y1, y2 = int(h*0.2), int(h*0.8)
        crop = eye_roi[y1:y2, :]
        
        # 2. SKIN REMOVAL (The "Isolation" Magic)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Skin Mask (Hue 0-25)
        lower_skin = np.array([0, 40, 60]); upper_skin = np.array([25, 255, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Invert -> "Not Skin" Mask (Sclera + Iris)
        eye_surface_mask = cv2.bitwise_not(skin_mask)
        
        # 3. VESSEL DETECTION (In the "Not Skin" area)
        # Red Hue 0-10 & 170-180 with High Saturation (>80)
        lower_red1 = np.array([0, 80, 50]); upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 80, 50]); upper_red2 = np.array([180, 255, 255])
        
        raw_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Valid Vessels = (Is Red) AND (Is NOT Skin)
        vessel_mask = cv2.bitwise_and(raw_red, raw_red, mask=eye_surface_mask)
        
        # 4. CALCULATE STATS
        red_pixels = cv2.countNonZero(vessel_mask)
        eye_area = cv2.countNonZero(eye_surface_mask)
        if eye_area == 0: eye_area = 1 # Avoid crash
        
        red_ratio = red_pixels / eye_area
        
        # Brightness (only on non-skin part)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        brightness = cv2.mean(gray, mask=eye_surface_mask)[0]
        
        # 5. DIAGNOSIS
        status = "Normal"
        if brightness < FATIGUE_THRESH:
            status = "Fatigue"
        elif red_ratio > REDNESS_THRESH:
            status = "Irritated"
            
        return {
            "status": status,
            "redness": red_ratio,
            "brightness": brightness,
            "crop": crop,
            "vessel_mask": vessel_mask
        }

    def process_image(self, img_path):
        frame = cv2.imread(img_path)
        if frame is None: return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- DETECTION PHASE ---
        detected_eyes = [] # Will store (x,y,w,h) relative to image
        
        # Plan A: Find Face -> Find Eyes inside
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        for (fx, fy, fw, fh) in faces:
            roi_gray = gray[fy:fy+fh, fx:fx+fw]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 6, minSize=(30, 30))
            for (ex, ey, ew, eh) in eyes:
                # Convert to global coordinates
                detected_eyes.append((fx+ex, fy+ey, ew, eh))
        
        # Plan B: If no face, search whole image
        if not detected_eyes:
            detected_eyes_rects = self.eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
            for (ex, ey, ew, eh) in detected_eyes_rects:
                detected_eyes.append((ex, ey, ew, eh))

        # SORT EYES: Left to Right (so Eye 1 is always left)
        detected_eyes.sort(key=lambda b: b[0])
        
        # --- ANALYSIS PHASE ---
        results = []
        
        for i, (x, y, w, h) in enumerate(detected_eyes):
            eye_roi = frame[y:y+h, x:x+w]
            
            # ISOLATE AND ANALYZE
            data = self.analyze_single_eye(eye_roi)
            
            # Add metadata
            data['box'] = (x, y, w, h)
            data['id'] = i + 1
            results.append(data)
            
            # Draw on Main Image
            color = (0, 255, 0)
            if data['status'] == "Irritated": color = (255, 0, 0)
            elif data['status'] == "Fatigue": color = (255, 255, 0)
            
            cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame_rgb, f"Eye {i+1}: {data['status']}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame_rgb, results

# --- VISUALIZATION ---
def show_full_report(orig_img, eye_results, filename):
    num_eyes = len(eye_results)
    if num_eyes == 0:
        print(f"❌ No eyes found in {filename}")
        return

    # Create Grid: 1 Row for Main Image + 1 Row PER EYE
    total_rows = 1 + num_eyes
    fig = plt.figure(figsize=(10, 4 * total_rows))
    
    # 1. Show Main Image (Top)
    ax_main = plt.subplot2grid((total_rows, 3), (0, 0), colspan=3)
    ax_main.imshow(orig_img)
    ax_main.set_title(f"Main Detection: {filename}")
    ax_main.axis('off')
    
    # 2. Show Each Eye Individually
    for i, res in enumerate(eye_results):
        row = i + 1
        
        # Eye Crop (RGB)
        ax_crop = plt.subplot2grid((total_rows, 3), (row, 0))
        ax_crop.imshow(cv2.cvtColor(res['crop'], cv2.COLOR_BGR2RGB))
        ax_crop.set_title(f"Eye {res['id']} (Isolated)")
        ax_crop.axis('off')
        
        # Vessel Mask (Black/White)
        ax_mask = plt.subplot2grid((total_rows, 3), (row, 1))
        ax_mask.imshow(res['vessel_mask'], cmap='gray')
        ax_mask.set_title("Vessels Only (Skin Removed)")
        ax_mask.axis('off')
        
        # Stats Box (Text)
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
    my_files = ['normal.jpg', 'red2.jpg', 'halfhalf.jpg'] 
    
    for f in my_files:
        print(f"\n--- Processing {f} ---")
        img_res, results = lab.process_image(f)
        
        if img_res is not None:
            # Print to Console
            for res in results:
                print(f"  👁️ Eye {res['id']}: [{res['status']}] (Redness: {res['redness']*100:.1f}%)")
            
            # Show Visual Report
            show_full_report(img_res, results, f)