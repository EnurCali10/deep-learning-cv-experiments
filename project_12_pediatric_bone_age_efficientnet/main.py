"""
problem tanimi: kemik yasi tahmini, regresyon problemi, insan eline ait farkli goruntuler goruntu isleme problemi

veri seti: https://www.kaggle.com/datasets/vaillant/rsna-pediatric-bone-age-challenge-n1200
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, concatenate, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.applications import EfficientNetB0

#veri setini yukleme ve temizleme 
df = pd.read_csv("train.csv")
print("Mevcut Sütunlar:", df.columns)

#klasordeki gorseli gercekten var olan resimleri al
# main.py dosyasının bilgisayarda veya sunucuda çalıştığı tam konumu otomatik bulur
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Bulunan konumun sonuna 'images' klasörünü ekler (Windows'ta \, Linux'ta / koyar)
image_folder = os.path.join(BASE_DIR, 'images')
avaliable_files = set(os.listdir(image_folder))
available_ids = set(f.replace (".png", "") for f in avaliable_files if f.endswith(".png"))

df = df[df["pid"].astype(str).isin(available_ids)].reset_index(drop=True)
df["boneage"] = df["bone_age"] / 240.0
df["path"] = df["pid"].apply(lambda x: os.path.join(image_folder, f"{x}.png"))
print(df.head())

#yas dagilimi
plt.hist(df["boneage"], bins=20)
plt.xlabel("Bone Age (normalized)")
plt.ylabel("Frequency")
plt.title("Distribution of Bone Age")
plt.tight_layout()
plt.show()

#goruntuleri okuma ve on isleme
def load_images(df, img_size=224):
    images = []
    valid_indices = []
    for i, path in enumerate(df["path"]):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("uyari")
            continue
        img= cv2.resize(img, (img_size, img_size))
        img = img / 255.0 #normalization
        img = np.stack([img, img, img], axis=-1)  # 3 kanala cevir (EfficientNet icin)
        images.append(img)
        valid_indices.append(i)
    new_df = df.iloc[valid_indices].reset_index(drop=True)
    return np.array(images), new_df["boneage"].values, new_df["female"].values.astype(np.float32)

X, y, gender = load_images(df)
print(X.shape)

#egitim ve test veri seti olusturma
X_train, X_val, y_train, y_val, g_train, g_val = train_test_split(X, y, gender, test_size=0.15, random_state=42)

#data augmentation 
datagen = ImageDataGenerator(
    horizontal_flip = True,
    zoom_range = 0.15,
    width_shift_range = 0.15,
    height_shift_range = 0.15,
    rotation_range = 10,
    brightness_range = [0.8, 1.2]
)

datagen.fit(X_train)

#cnn modeli - transfer learning: EfficientNetB0 tabanli
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # dondurulmus kalsin, fine-tune yok

img_input = base_model.input
gender_input = Input(shape=(1,))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)

# cinsiyet bilgisini modele dahil et
combined = concatenate([x, gender_input])
x = Dense(128, activation="relu")(combined)
x = Dropout(0.3)(x)
output = Dense(1, activation="linear")(x) #regresyon ciktisi

model = Model(inputs=[img_input, gender_input], outputs=output)

#model compile
model.compile(
    optimizer = Adam(learning_rate = 0.001),
    loss = "mae",
    metrics = [MeanAbsoluteError()]
)

#callback tanimlama: erken durdurma, model kaydi, LR ayarlamasi
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True, monitor= "val_loss"),
    ModelCheckpoint("bone_age_model.keras", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(patience= 5, factor= 0.5,monitor= "val_loss" )
]

#model egitimi
class BoneAgeGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, g, datagen=None, batch_size=32, **kwargs):
        super().__init__(**kwargs)
        self.X = X
        self.y = y
        self.g = g
        self.datagen = datagen
        self.batch_size = batch_size

    def __len__(self):
        return len(self.X) // self.batch_size

    def __getitem__(self, idx):
        X_batch = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        y_batch = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        g_batch = self.g[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.datagen:
            X_batch = next(self.datagen.flow(X_batch, batch_size=self.batch_size, shuffle=False))
        return (X_batch, g_batch), y_batch  # liste yerine tuple

train_gen = BoneAgeGenerator(X_train, y_train, g_train, datagen=datagen, batch_size=32)

# sadece ust katmanlar egitiliyor, fine-tune yok
print("Egitim basliyor...")
history = model.fit(
    train_gen,
    validation_data=([X_val, g_val], y_val),
    epochs=50,
    callbacks=callbacks
)

#model degerlendirme
plt.figure()
plt.plot(history.history["loss"], label = "training mae")
plt.plot(history.history["val_loss"], label = "validation mae")
plt.xlabel("epochs")
plt.ylabel("MAE")
plt.title("training performance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

preds = model.predict([X_val, g_val])*240
actuals = y_val* 240

plt.figure(figsize=(12, 6))
for i in range (10):
    plt.subplot(2,5,i+1)
    # Görüntünün üçüncü boyutunu (kanal) squeeze() ile kaldırarak doğru boyuta indirgedik.
    plt.imshow(X_val[i][:,:,0], cmap= "gray")
    plt.title(f"Tahmin: {preds[i][0]:.0f}\nGercek:{actuals[i]:.0f}")
    plt.axis("off")

plt.suptitle("kemik yasi tahmin sonuclari")
plt.tight_layout()
plt.show()







