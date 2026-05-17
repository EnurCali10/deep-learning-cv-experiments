# Heuristic Facial Emotion Tracker via MediaPipe

## Proje Özeti
Bu proje, web kamerası üzerinden gerçek zamanlı insan yüzü tespiti yapan ve geometrik kurallara (heuristics) dayanarak temel duygu durumlarını (Mutlu, Şaşkın, Nötr) tahmin eden bir bilgisayarlı görü prototipidir. 

Sistem, makine öğrenimi tabanlı bir duygu sınıflandırıcı (classifier) kullanmak yerine, Google MediaPipe Face Landmarker modelinin sunduğu yüz eklem noktaları (landmarks) arasındaki Öklid mesafelerini hesaplayarak eşik tabanlı bir çıkarım yapmaktadır.

## Kullanılan Teknolojiler ve Yöntem
* **Yüz Tespiti ve Landmark Çıkarımı:** MediaPipe Tasks Vision (`FaceLandmarker`)
* **Görüntü İşleme ve Akış:** OpenCV
* **Matematiksel Hesaplama:** NumPy (Euclidean Distance / `np.linalg.norm`)
* **Algoritma Mantığı:**
    * *Şaşkın:* Kaş (65) ve Göz (159) arasındaki dikey mesafenin artışı.
    * *Mutlu:* Sol Dudak (61) ve Sağ Dudak (291) arasındaki yatay mesafenin artışı.

## Kurulum ve Gereksinimler
Sistemi çalıştırmak için aşağıdaki Python kütüphanelerini kurun:
```bash
pip install opencv-python mediapipe numpy
