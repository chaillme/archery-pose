import cv2
import time
import os
import json
import numpy as np
import mediapipe as mp

# ---- CONFIG ----
with open('config.json', 'r') as f:
    config = json.load(f)

model_path = 'pose_landmarker_lite.task'

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.arccos(np.clip(cos, -1.0, 1.0)) * 180 / np.pi

def play_sound(last_time, cooldown):
    now = time.time()
    if now - last_time > cooldown:
        sound_path = config['sound_file']
        os.system(f"aplay '{sound_path}' &> /dev/null 2>&1 || "
                  f"aplay /usr/share/sounds/alsa/Front_Left.wav &> /dev/null 2>&1 || "
                  f"printf '\\a'")
        return now
    return last_time

# ---- LANDMARKER ----
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=config['min_detection_confidence'],
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    num_poses=1,
)

# ---- WEBCAM depuis CONFIG ✅ ----
webcam_id = config.get('webcam_id', 0)  # 0=défaut, 2=USB, etc.
cap = cv2.VideoCapture(webcam_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(f"🚀 Webcam ID: {webcam_id}")

# ---- PLEIN ÉCRAN ----
cv2.namedWindow("Tir a l'arc - Pose", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Tir a l'arc - Pose", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

last_sound_time = 0
sound_cooldown = 2.0

with PoseLandmarker.create_from_options(options) as landmarker:
    prev_timestamp_ms = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= prev_timestamp_ms:
            timestamp_ms = prev_timestamp_ms + 1
        prev_timestamp_ms = timestamp_ms

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        status = "INCORRECTE"
        color = (0, 0, 255)
        h, w, _ = frame.shape

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            # ---- POINTS VERTS ----
            for lm in landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

            # ---- LIGNES BLEUES archery ----
            archery_connections = [
                (11, 12), (12, 24), (11, 23),  # torse
                (12, 14), (14, 16),  # bras arc DROIT
                (11, 13), (13, 15),  # bras tir GAUCHE
            ]
            
            for conn in archery_connections:
                lm1, lm2 = landmarks[conn[0]], landmarks[conn[1]]
                x1, y1 = int(lm1.x * w), int(lm1.y * h)
                x2, y2 = int(lm2.x * w), int(lm2.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # ---- ANALYSE POSTURE ----
            shoulder_angle = calculate_angle(landmarks[11], landmarks[12], landmarks[24])
            bow_arm_angle = calculate_angle(landmarks[12], landmarks[14], landmarks[16])
            elbow_raise_offset = landmarks[14].y - landmarks[12].y

            bow_arm_min = config.get('bow_arm_angle_min', 160)
            elbow_max_offset = config.get('elbow_raise_offset', 0.02)

            is_correct = (
                config['shoulder_angle_min'] <= shoulder_angle <= config['shoulder_angle_max'] and
                bow_arm_angle >= bow_arm_min and
                elbow_raise_offset < elbow_max_offset
            )

            if is_correct:
                status = "CORRECTE!"
                color = (0, 255, 0)
                last_sound_time = play_sound(last_sound_time, sound_cooldown)

            # ---- AFFICHAGE ----
            cv2.putText(frame, f"Posture: {status}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"Webcam: {webcam_id} | Epaules: {shoulder_angle:.0f}°", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Bras arc: {bow_arm_angle:.0f}° (>={bow_arm_min}°)", 
                       (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Coude: {'RELEVÉ' if elbow_raise_offset < elbow_max_offset else 'BAS'}", 
                       (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, "Q/ESC=Quitter", (20, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("Tir a l'arc - Pose", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

cv2.destroyAllWindows()
cap.release()
