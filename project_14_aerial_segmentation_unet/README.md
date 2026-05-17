# Aerial Image Semantic Segmentation via U-Net

## Proje Özeti
Bu proje, yüksek çözünürlüklü uydu ve hava görüntüleri (aerial imagery) üzerinde anlamsal segmentasyon (semantic segmentation) yapmak amacıyla geliştirilmiş bir **U-Net** mimarisi uygulamasıdır. 

Sistem, encoder-decoder yapısı ve atlama bağlantıları (skip connections) kullanarak mekansal (spatial) bilgi kaybını en aza indirmekte ve havadan çekilmiş görüntüleri ikili (binary) maskeler halinde segmente etmektedir. 

## Model Mimarisi ve Hiperparametreler
* **Mimari:** Standart U-Net (4 seviyeli Encoder-Decoder + Bottleneck)
* **Girdi Boyutu:** 128x128x3 (RGB)
* **Çıktı Boyutu:** 128x128x1 (Sigmoid aktivasyonlu ikili maske)
* **Kayıp Fonksiyonu:** Binary Crossentropy
* **Optimizasyon:** Adam
* **Geri Çağrılar (Callbacks):** * `ModelCheckpoint` (En iyi ağırlıkları kaydetmek için)
    * `ReduceLROnPlateau` (Öğrenme oranını dinamik düşürmek için)
    * `EarlyStopping` (Aşırı öğrenmeyi -overfitting- engellemek için)

## Kurulum ve Gereksinimler
Projenin bağımlılıklarını kurmak için izole bir Python ortamı kullanın:

```bash
pip install tensorflow keras numpy opencv-python matplotlib scikit-learn