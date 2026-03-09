import cv2

print("Tes webcams disponibles :")
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Webcam {i} OK")
        cap.release()
    else:
        print(f"❌ Webcam {i} indisponible")
