# Project 8: Multi-Object Vehicle Tracking with YOLOv8 & ByteTrack

Bu çalışma, trafik güvenliği ve akış analizi için kritik olan araçların (araba, kamyon, otobüs) hareket halindeyken tespit edilmesini ve her birine özel bir ID atanarak takip edilmesini içerir.

##  Veri Kaynağı
- **İçerik:** Gerçek trafik kameralarından alınan, yüksek varyasyonlu araç görüntüleri ve videoları.



##  Teknik Mimari
- **Dedektör:** YOLOv8m (Medium) - Küçük araçları ve uzak nesneleri daha iyi yakalamak için tercih edilmiştir.
- **Takip Algoritması:** ByteTrack (Persist=True) - Nesne kaybını (occlusion) minimize eden düşük güvenli kutuları da hesaba katan gelişmiş bir tracker.
- **Performans:** NVIDIA RTX 3050 üzerinde optimize edilerek saniyede yüksek kare (FPS) işleme hızı elde edilmiştir.

##  Green AI ve Verimlilik
Kaggle'dan alınan ham veriler, modelin sadece gerekli öznitelikleri (features) öğreneceği şekilde ön işleme tabi tutulmuştur. Bu sayede:
- Gereksiz GPU çevrimleri engellenmiş,
- **RTX 3050**'nin güç tüketimi (TDP) kontrol altında tutularak sürdürülebilir bir eğitim süreci sağlanmıştır.

##  Kurulum
```bash
pip install ultralytics opencv-python
python main.py