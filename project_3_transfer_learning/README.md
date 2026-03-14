# Project 3 — Transfer Learning ile Zatürre Tespiti

## Açıklama

DenseNet121 modelini kullanarak göğüs röntgeni görüntülerinden zatürre (PNEUMONIA) tespiti yapar.  
Transfer learning ile önceden ImageNet üzerinde eğitilmiş ağırlıklar kullanılır.

## Model Mimarisi

```
DenseNet121 (ImageNet ağırlıkları, dondurulmuş)
→ GlobalAveragePooling2D
→ Dense(128, ReLU)
→ Dropout(0.5)
→ Dense(1, Sigmoid)   ← Binary: NORMAL / PNEUMONIA
```

## Veri Seti

Kaggle — Chest X-Ray Images (Pneumonia)  
🔗https://www.kaggle.com/datasets/alifrahman/chestxraydataset

İndirdikten sonra `DATA_DIR` değişkenini güncelleyin:

```python
DATA_DIR = r"veri_seti_yolunuz/chest_xray"
```

Klasör yapısı şu şekilde olmalı:
```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Kullanım

```bash
python transfer_learning.py
```

## Değerlendirme

Eğitim sonunda test seti üzerinde Confusion Matrix görselleştirilir.

## Callbacks

- **EarlyStopping** — val_loss 3 epoch iyileşmezse dur  
- **ReduceLROnPlateau** — öğrenme oranını otomatik düşür  
- **ModelCheckpoint** — en iyi modeli `best_model.h5` olarak kaydet  
