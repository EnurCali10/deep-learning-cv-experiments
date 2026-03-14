# Project 1 — ANN ile MNIST Rakam Sınıflandırma

## Açıklama

MNIST veri setindeki el yazısı rakamları (0–9) ANN (Yapay Sinir Ağı) ile sınıflandırır.  
Görüntüler önce OpenCV ile işlenir, ardından model eğitilir.

## Görüntü İşleme Aşamaları

1. **Histogram Eşitleme** — Kontrast iyileştirme  
2. **Gaussian Blur** — Gürültü azaltma  
3. **Canny Edge Detection** — Kenar tespiti  

## Model Mimarisi

```
Input (784) → Dense(128, ReLU) → Dropout(0.5) → Dense(64, ReLU) → Dense(10, Softmax)
```

## Veri Seti

MNIST — TensorFlow/Keras üzerinden otomatik indirilir.

```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

## Kullanım

```bash
python ann.py
```

Kendi el yazısı rakamınızı test etmek için `ann.py` dosyasının en altındaki satırı güncelleyin:

```python
kağıttan_tahmin_et("resim_yolunuz.jpg", model)
```

## Sonuçlar

- Eğitim: 10.000 örnek, Test: 2.000 örnek
- 20 epoch, batch size 32
