import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    """
    Üç nokta arasındaki açıyı derece cinsinden hesaplar.
    """
    a = np.array(a) # Birinci nokta
    b = np.array(b) # İkinci nokta (eklem noktası)
    c = np.array(c) # Üçüncü nokta
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# MediaPipe modülleri
mp_drawing = mp.solutions.drawing_utils 
mp_pose = mp.solutions.pose 

cap = cv2.VideoCapture("squat_test1.avi")

if not cap.isOpened():
    print("Video dosyası bulunamadı veya açılamadı. Dosya adını kontrol edin.")

# Sayaç ve aşama değişkenleri
counter = 0 
stage = None 

def classify_pose(knee_angle):
    """
    Diz açısına göre poz sınıflandırma.
    """
    if knee_angle > 150:
        return "Standing"
    elif 100 <= knee_angle <= 150:
        return "Lunging / Bending"
    else:
        return "Squatting"

# Pose modülünü oluştur
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret: # Video bittiğinde döngüden çık
            break
        
        # Görüntüyü RGB formatına çevir
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        results = pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Sağ kalça, diz, ayak bileği eklemi için koordinatları al
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Diz açısını hesapla
            knee_angle = calculate_angle(hip, knee, ankle)
            
            # Poz sınıflandır
            pose_classification = classify_pose(knee_angle)

            # Squat sayacı ve aşama mantığı
            if knee_angle < 90:
                stage = "down"
            if knee_angle > 160 and stage == "down":
                stage = "up"
                counter += 1
            
            # Açı, sınıf ve sayacı görüntüle
            cv2.putText(image, f'diz aci: {int(knee_angle)}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(image, f'Pose: {pose_classification}', (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(image, f'squat sayisi: {counter}', (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        except AttributeError:
            pass # Land (nokta) algılanmadığında duraksamayı engelle
            
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        
        cv2.imshow('Pose Estimation', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()