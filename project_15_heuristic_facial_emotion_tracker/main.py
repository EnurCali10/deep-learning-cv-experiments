"""
problem tanimi: web cam ile insan yuzunu alalim duygu tanima: mutlu,notr,sasirmis
veri seti: kendi goruntu verilerimizi kullanacagiz


"""


import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. MediaPipe Tasks Yapılandırması
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,  # Bu kalsın, ileride daha gelişmiş duygu tanıma için lazım olur
    num_faces=1,
    running_mode=vision.RunningMode.VIDEO
)

# Dedektörü başlat
detector = vision.FaceLandmarker.create_from_options(options)

def detect_emotion(landmarks, image_width, image_height):
    def get_point(index):
        lm = landmarks[index]
        return np.array([lm.x * image_width, lm.y * image_height])

    # Duygu tespiti için landmark indexleri 
    # Kaş (65) ve Göz (159)
    brow_point = get_point(65)
    eye_point = get_point(159)
    brow_lift = np.linalg.norm(brow_point - eye_point)

    # Dudak sol (61) ve sağ (291)
    mouth_left = get_point(61)
    mouth_right = get_point(291)
    mouth_width = np.linalg.norm(mouth_left - mouth_right)

    # Basit eşik değerleri (Kameraya uzaklığına göre bunları ayarla)
    if brow_lift > 20:
        return "Saskin"
    elif mouth_width > 50:
        return "Mutlu"
    else:
        return "Notr"

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # RGB'ye çevir (MediaPipe gereksinimi)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Zaman damgası (VIDEO modu için gerekli)
    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    
    # Yüz tespiti yap
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)

    h, w, _ = frame.shape

    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            # Duygu tahmini
            emotion = detect_emotion(face_landmarks, w, h)
            
            # Ekrana yazdır
            cv2.putText(frame, f"Duygu: {emotion}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Landmarkları çiz 
            for lm in face_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("Duygu Tanima - MediaPipe Tasks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()