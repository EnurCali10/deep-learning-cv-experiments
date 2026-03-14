# Project 2 — CNN ile Çiçek Sınıflandırma

## Açıklama

TensorFlow Flowers veri setindeki 5 çiçek türünü CNN (Evrişimli Sinir Ağı) ile sınıflandırır.

## Model Mimarisi

```
Input (180x180x3)
→ Conv2D(32) + MaxPool
→ Conv2D(64) + MaxPool
→ Conv2D(128) + MaxPool
→ Flatten → Dense(128) → Dropout(0.5) → Dense(5, Softmax)
```

## Veri Seti

TF Flowers — `tensorflow-datasets` üzerinden otomatik indirilir.

```python
from tensorflow_datasets import load
(ds_train, ds_val), ds_info = load('tf_flowers', split=['train[:80%]', 'train[80%:]'], ...)
```

## Data Augmentation

Eğitim verisine uygulanan dönüşümler: yatay çevirme, parlaklık, kontrast, rastgele kırpma.

## Kullanım

```bash
python cnn.py
```

## Callbacks

- **EarlyStopping** — val_loss 3 epoch iyileşmezse dur  
- **ReduceLROnPlateau** — öğrenme oranını otomatik düşür  
- **ModelCheckpoint** — en iyi modeli `best_model.h5` olarak kaydet  
