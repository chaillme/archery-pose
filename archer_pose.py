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

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cos_value = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_value = np.clip(cos_value, -1.0, 1.0)

    return np.degrees(np.arccos(cos_value))


def play_sound(last_time, cooldown):
    now = time.time()
    if now - last_time > cooldown:
        sound_path = config.get('sound_file', '')
        os.system(
            f"aplay '{sound_path}' &> /dev/null 2>&1 || "
            f"aplay /usr/share/sounds/alsa/Front_Left.wav &> /dev/null 2>&1 || "
            f"printf '\\a'"
        )
        return now
    return last_time


def get_archer_side_indices(handedness):
    """
    handedness = 'right' => archer droitier
      - bras d'arc = bras gauche
      - bras de traction = bras droit

    handedness = 'left' => archer gaucher
      - bras d'arc = bras droit
      - bras de traction = bras gauche

    Indices MediaPipe Pose:
      11 = épaule gauche
      12 = épaule droite
      13 = coude gauche
      14 = coude droit
      15 = poignet gauche
      16 = poignet droit
      23 = hanche gauche
      24 = hanche droite
    """
    if handedness == "right":
        # Archer droitier : arc tenu à gauche, traction à droite
        bow_shoulder, bow_elbow, bow_wrist = 11, 13, 15
        rear_shoulder, rear_elbow, rear_wrist = 12, 14, 16
    else:
        # Archer gaucher : arc tenu à droite, traction à gauche
        bow_shoulder, bow_elbow, bow_wrist = 12, 14, 16
        rear_shoulder, rear_elbow, rear_wrist = 11, 13, 15

    return bow_shoulder, bow_elbow, bow_wrist, rear_shoulder, rear_elbow, rear_wrist


# ---- PARAMÈTRES CONFIG ----
handedness = config.get("handedness", "right").lower()
if handedness not in ["right", "left"]:
    handedness = "right"

bow_arm_min = config.get('bow_arm_angle_min', 160)
rear_elbow_max_offset = config.get('rear_elbow_max_offset', 0.02)
rear_arm_angle_min = config.get('rear_arm_angle_min', 70)
rear_arm_angle_max = config.get('rear_arm_angle_max', 140)
shoulder_angle_min = config.get('shoulder_angle_min', 70)
shoulder_angle_max = config.get('shoulder_angle_max', 110)

# ---- LANDMARKER ----
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=config.get('min_detection_confidence', 0.5),
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    num_poses=1,
)

# ---- WEBCAM ----
webcam_id = config.get('webcam_id', 0)
cap = cv2.VideoCapture(webcam_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print(f"🚀 Webcam ID: {webcam_id}")
print(f"🏹 Archer: {'DROITIER' if handedness == 'right' else 'GAUCHER'}")

# ---- PLEIN ÉCRAN ----
window_name = "Tir a l'arc - Pose"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

        h, w, _ = frame.shape
        status = "AUCUNE POSE"
        color = (0, 0, 255)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            bow_shoulder, bow_elbow, bow_wrist, rear_shoulder, rear_elbow, rear_wrist = \
                get_archer_side_indices(handedness)

            # ---- POINTS VERTS ----
            for lm in landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)

            # ---- LIGNES BLEUES ----
            archery_connections = [
                (11, 12),  # ligne d'épaules
                (11, 23), (12, 24),  # torse
                (bow_shoulder, bow_elbow), (bow_elbow, bow_wrist),  # bras d'arc
                (rear_shoulder, rear_elbow), (rear_elbow, rear_wrist),  # bras arrière
            ]

            for conn in archery_connections:
                lm1, lm2 = landmarks[conn[0]], landmarks[conn[1]]
                x1, y1 = int(lm1.x * w), int(lm1.y * h)
                x2, y2 = int(lm2.x * w), int(lm2.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # ---- ANALYSE POSTURE ----
            # Angle buste/épaule conservé dans l'esprit de ton code initial
            shoulder_angle = calculate_angle(landmarks[11], landmarks[12], landmarks[24])

            # Bras d'arc : doit être bien tendu
            bow_arm_angle = calculate_angle(
                landmarks[bow_shoulder],
                landmarks[bow_elbow],
                landmarks[bow_wrist]
            )

            # Bras arrière : angle de traction
            rear_arm_angle = calculate_angle(
                landmarks[rear_shoulder],
                landmarks[rear_elbow],
                landmarks[rear_wrist]
            )

            # Coude arrière : doit être au niveau ou légèrement au-dessus de l'épaule
            rear_elbow_offset = landmarks[rear_elbow].y - landmarks[rear_shoulder].y

            is_shoulder_ok = shoulder_angle_min <= shoulder_angle <= shoulder_angle_max
            is_bow_arm_ok = bow_arm_angle >= bow_arm_min
            is_rear_elbow_ok = rear_elbow_offset <= rear_elbow_max_offset
            is_rear_arm_ok = rear_arm_angle_min <= rear_arm_angle <= rear_arm_angle_max

            errors = []

            if not is_shoulder_ok:
                errors.append("epaules")
            if not is_bow_arm_ok:
                errors.append("bras d'arc")
            if not is_rear_elbow_ok:
                errors.append("coude arriere")
            if not is_rear_arm_ok:
                errors.append("bras arriere")

            if not errors:
                status = "CORRECTE !"
                color = (0, 255, 0)
                last_sound_time = play_sound(last_sound_time, sound_cooldown)
            else:
                status = "INCORRECTE: " + ", ".join(errors)
                color = (0, 0, 255)

            # ---- AFFICHAGE ----
            archer_label = "DROITIER" if handedness == "right" else "GAUCHER"

            cv2.putText(frame, f"Archer: {archer_label}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, f"Posture: {status}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

            cv2.putText(frame, f"Epaules: {shoulder_angle:.0f}° [{shoulder_angle_min}-{shoulder_angle_max}]",
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if is_shoulder_ok else (0, 0, 255), 2)

            cv2.putText(frame, f"Bras d'arc: {bow_arm_angle:.0f}° (>={bow_arm_min}°)",
                        (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if is_bow_arm_ok else (0, 0, 255), 2)

            cv2.putText(frame, f"Coude arriere: {'RELEVE' if is_rear_elbow_ok else 'TROP BAS'} ({rear_elbow_offset:.3f})",
                        (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if is_rear_elbow_ok else (0, 0, 255), 2)

            cv2.putText(frame, f"Bras arriere: {rear_arm_angle:.0f}° [{rear_arm_angle_min}-{rear_arm_angle_max}]",
                        (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if is_rear_arm_ok else (0, 0, 255), 2)

            cv2.putText(frame, f"Webcam: {webcam_id}", (20, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, "Q / ESC = Quitter", (20, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

cv2.destroyAllWindows()
cap.release()
