"""
problem tanimi: unet kullanarak uydu görüntüleri ile segmentasyon
unet: encoder-decoder mimarisi, skip connections, convolutional layers
az veri ile iyi performans: transfer learning, data augmentation, regularization
veri seti: uydu görüntüleri ve segmentasyon maskeleri
veri ön işleme: normalize etme, veri artırma (data augmentation)
model eğitimi: kayıp fonksiyonu (loss function), optimizasyon algoritması, eğitim döngüsü
model değerlendirme: doğruluk, IoU (Intersection over Union), F1 skoru
"""
import os   
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

#veri hazırlama
def load_dataset(root, img_size=(128, 128)):
    images = []#goruntu bos listesi
    masks = []#mask bos listesi
    for tile in sorted(os.listdir(root)): # her bir tile klasorunu sirasiyle dolas
        img_dir = os.path.join(root, tile, 'images') #goruntu klasorunun yolu, 
        mask_dir = os.path.join(root, tile, 'masks') #mask klasorunun yolu ,
        if not os.path.exists(img_dir): continue #goruntu veya mask klasoru yoksa atla
        for f in os.listdir(img_dir): #goruntu klasorundeki her bir dosyayi dolas
            if not f.lower().endswith( '.jpg'): continue #gsadece jpg dosyalarini isleme al
            img_path = os.path.join(img_dir, f) #goruntu dosya yolunu elde ettik
            mask_path = os.path.join(mask_dir, os.path.splitext(f)[0]+".png") #mask dosyasinin tam yolu
            if not os.path.exists(mask_path): continue 
            #goruntuyu oku ve rgbye cevir
            img= cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img= cv2.resize(img, img_size)/255.#goruntuyu boyutlandir ve normalize et

            #maskeyi gri tonlamada oku ve yeniden boyutlandir ve normalize et 
            mask=cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask=cv2.resize(mask, img_size)
            mask = np.expand_dims(mask, axis=-1)/255.0 #maskeyi tek kanallı yap ve normalize et

            images.append(img) #goruntuyu listeye ekle
            masks.append(mask) #maskeyi listeye ekle
    return np.array(images, dtype="float32"), np.array(masks, dtype="float32") #goruntu ve maskeleri numpy dizilerine cevir

x, y = load_dataset("aerial_dataset", img_size=(128, 128)) 

print(f"Dataset loaded: {x.shape} images, {y.shape} masks")

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"Training set: {x_train.shape} images, {y_train.shape} masks")
print(f"Validation set: {x_val.shape} images, {y_val.shape} masks")


#unet mimarisi tanimlama
def unet_model(input_size=(128, 128, 3)):
    inputs = keras.Input(input_size)#girdi katmanı

    # Encoder: feature extraction ve downsampling
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)#16 filtre 3x3 kernel
    c1 = layers.Conv2D(16,3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)#downsampling 64x64

    c2 = layers.Conv2D(32,3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3 ,activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D()(c3)

    c4 = layers.Conv2D(128,3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D()(c4)

    # Bottleneck: en derin seviye 
    c5 = layers.Conv2D(256,3, activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(c5)

    # Decoder: upsampling ve skip connections
    u6 = layers.Conv2DTranspose(128, 2, strides=(2,2), padding="same") (c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(64, 2, strides=(2,2), padding="same")(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64,3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(32, 2, strides=(2,2), padding="same")(c7)   
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, 3, activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(32, 3, activation='relu', padding='same')(c8)   

    u9 = layers.Conv2DTranspose(16, 2, strides=(2,2), padding="same")(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(16, 3, activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(16, 3, activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9) #1x1 konvolusyon ile tek kanallı çıktı   
    return keras.Model(inputs, outputs)

#modeli eğitme
unet_model = unet_model()
unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

#callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint("unet_best_model.h5", save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(),
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
]

history=unet_model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10, batch_size=16, 
    callbacks=callbacks
)

#sonuçların değerlendirmesi
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

def show_predictions(idx=0):
    img = x_val[idx]
    mask = y_val[idx].squeeze()
    pred_raw = unet_model.predict(img[None, ...])[0].squeeze()
    mask_pred = (pred_raw > 0.5).astype("float32")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("True Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")  

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(mask_pred, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()  

show_predictions(1)
