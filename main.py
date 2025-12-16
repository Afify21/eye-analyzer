import cv2
import numpy as np
import os

# --- CONFIGURATION ---
REDNESS_THRESHOLD = 0.5  # Lowered slightly to be more sensitive
FATIGUE_THRESHOLD = 90     # Raised slightly (brightness < 90 = fatigue)
SYMMETRY_TOLERANCE = 0.15 
# ---------------------

class EyeAnalyzer:
    def __init__(self):
        # 1. Verify Haar Paths (Debug Step)
        face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        eye_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        
        print(f"Looking for Face XML at: {face_path}")
        print(f"Looking for Eye XML at: {eye_path}")
        
        # 2. Load Classifiers
        self.face_cascade = cv2.CascadeClassifier(face_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_path)
        
        # Check if they loaded
        if self.face_cascade.empty():
            print("❌ ERROR: Face Cascade failed to load! Check opencv install.")
        if self.eye_cascade.empty():
            print("❌ ERROR: Eye Cascade failed to load!")

    def analyze_eye_region(self, eye_roi):
        # [cite_start]FATIGUE (Avg Brightness) [cite: 55]
        gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray_eye)
        
        # [cite_start]REDNESS (HSV) [cite: 41, 44]
        hsv_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2HSV)
        
        # [cite_start]Red Masks [cite: 45]
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv_eye, lower_red1, upper_red1) + cv2.inRange(hsv_eye, lower_red2, upper_red2)
        
        # [cite_start]Ratio [cite: 47]
        red_pixels = cv2.countNonZero(mask)
        total_pixels = eye_roi.shape[0] * eye_roi.shape[1]
        redness_ratio = red_pixels / total_pixels if total_pixels > 0 else 0
        
        return avg_brightness, redness_ratio

    def run(self):
        # Try Index 0 first, if fails try 1 (common mac issue)
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
             cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
        
        if not cap.isOpened():
            print("❌ Error: Could not open ANY camera.")
            return

        print("✅ Camera active. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret: break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # --- TWEAKED DETECTION SETTINGS ---
            # scaleFactor=1.1 (Checks more scales, slower but catches more faces)
            # minNeighbors=4 (Less strict, 3 might cause glitches, 5 is too strict)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            # DEBUG: Print number of faces found
            # print(f"Faces Detected: {len(faces)}") 

            for (x, y, w, h) in faces:
                # Draw Blue Box around Face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                # Detect Eyes (Relaxed settings for eyes too)
                eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4)
                
                eye_data = []

                for (ex, ey, ew, eh) in eyes:
                    eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
                    brightness, redness = self.analyze_eye_region(eye_roi)
                    eye_data.append((brightness, redness))
                    
                    # Logic
                    status = "Normal"
                    box_color = (0, 255, 0)
                    
                    if redness > REDNESS_THRESHOLD:
                        status = "Irritated"
                        box_color = (0, 0, 255)
                    elif brightness < FATIGUE_THRESHOLD:
                        status = "Fatigue"
                        box_color = (0, 255, 255)

                    # Draw Eye Box
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), box_color, 2)
                    cv2.putText(roi_color, status, (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

                # [cite_start]Symmetry Check (Only if 2 eyes found) [cite: 61]
                if len(eye_data) == 2:
                    diff_red = abs(eye_data[0][1] - eye_data[1][1])
                    if diff_red > SYMMETRY_TOLERANCE:
                         cv2.putText(frame, "SYMMETRY ALERT", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Name Tag
                cv2.putText(frame, "Youssef Afify", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Eye State Analyzer', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = EyeAnalyzer()
    app.run()