# Project 8: Multi-Object Vehicle Tracking with YOLOv8 & ByteTrack

Bu çalışma, trafik akışındaki araçların (araba, kamyon, otobüs vb.) gerçek zamanlı olarak algılanmasını, sınıflandırılmasını ve her bir araca özel bir ID atanarak takip edilmesini kapsar.

## Teknik Yaklaşım
Proje iki temel aşamadan oluşmaktadır:
1. [cite_start]**Detection (Tespit):** YOLOv8m (Medium) modeli kullanılarak araçların koordinatları ve sınıfları belirlenir.
2. [cite_start]**Tracking (Takip):** `ByteTrack` algoritması ve `persist=True` parametresi ile nesnelerin kareler arasındaki hareket sürekliliği sağlanır.


##  Parametre Analizi
- [cite_start]**Model:** YOLOv8m (Doğruluk ve hız dengesi için orta boy model seçilmiştir).
- [cite_start]**Güven Skoru (Confidence):** 0.3 (Daha küçük nesneleri kaçırmamak için optimize edilmiştir).
- [cite_start]**IOU Eşiği:** 0.5 (Kesişen araçların doğru kimliklendirilmesi için belirlenmiştir).
- [cite_start]**Tracker:** ByteTrack (Karmaşık sahnelerde yüksek performans sunar).

##  Green AI 
- **VRAM Optimizasyonu:** Orta ölçekli YOLOv8m modeli, **RTX 3050** donanımında aşırı ısınmaya ve yüksek enerji tüketimine yol açmadan akıcı çalışacak şekilde yapılandırılmıştır.
- **Hız:** Proje, yüksek FPS değerlerine odaklanarak GPU'nun birim kare başına harcadığı enerjiyi optimize eder.


##  Kurulum ve Kullanım
```bash
pip install ultralytics opencv-python
python main.py Kurulum ve Kullanım
```bash
pip install ultralytics opencv-python
python main.py