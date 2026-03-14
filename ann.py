"""
Mnist veri seti: 0-9 toplam 10 sinif var 
28x28 piksel boyutu
siyah beyaz resimler
60000 egitim, 10000 test
amac: ann ile resimleri tanila ve siniflandir

image processing:
histogram esitleme: kontrast iyilestirme
gaussian blur: gürültü azaltma
canny edge detection: kenar tespiti

ANN ile mnist veri seti siniflandirma
kütüphaneler:
tensorflow: keras ile ann modeli olusturma ve egitim
matplotlib: resimleri gorsellestirme
cv2 : opencv image processing 
"""

import cv2
import numpy as np 
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
#veriseti yükleme
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

#image processing
img = x_train[0]

stages = {"Orijinal": img}

eq    = cv2.equalizeHist(img)
stages["Histogram Esitleme"] = eq

blur  = cv2.GaussianBlur(eq, (5, 5), 0)
stages["Gaussian Blur"] = blur

edges = cv2.Canny(blur, 50, 150)
stages["Canny Kenarlari"] = edges

fig, axes = plt.subplots(2, 2, figsize=(6, 6))
axes = axes.flat
for ax, (title, im) in zip(axes, stages.items()):
    ax.imshow(im, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
plt.suptitle("MNIST Image Processing Stages")
plt.tight_layout()
plt.show()        

#preprocessing
def preprocess_image(image):
    img_eq    = cv2.equalizeHist(image)
    img_blur  = cv2.GaussianBlur(img_eq, (5, 5), 0)
    img_edges = cv2.Canny(img_blur, 50, 150)
    features  = img_edges.flatten() / 255.0
    return features

num_train = 10000
num_test  = 2000

x_train_processed = np.array([preprocess_image(img) for img in x_train[:num_train]])
y_train_sub       = y_train[:num_train]

x_test_processed  = np.array([preprocess_image(img) for img in x_test[:num_test]])
y_test_sub        = y_test[:num_test]

print(f"İşlenmiş X_train boyutu: {x_train_processed.shape}")
print(f"İşlenmiş X_test  boyutu: {x_test_processed.shape}")

# label one hot encode
# Softmax + categorical_crossentropy için gerekli
y_train_cat = to_categorical(y_train_sub, num_classes=10)
y_test_cat  = to_categorical(y_test_sub,  num_classes=10)

#ann model olusturma
model = Sequential([
    Dense(128, activation="relu", input_shape=(784,)),
    Dropout(0.5),               
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",    # one-hot label kullandığımız için
    metrics=["accuracy"]
)

model.summary()

#model egitim
history = model.fit(
    x_train_processed, y_train_cat,
    epochs=20,
    batch_size=32,
    validation_split=0.1,       # eğitim setinin %10'u validasyon
    verbose=1
)
#modelin degerlendirilmesi
test_loss, test_acc = model.evaluate(x_test_processed, y_test_cat, verbose=0)
print(f"\nTest Loss    : {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"],     label="Train Acc")
axes[0].plot(history.history["val_accuracy"], label="Val Acc")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["loss"],     label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Val Loss")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.suptitle("ANN Eğitim Grafikleri")
plt.tight_layout()
plt.show()

#model tahminleri
predictions = model.predict(x_test_processed[:10])
pred_labels = np.argmax(predictions, axis=1)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flat
for i, ax in enumerate(axes):
    ax.imshow(x_test[i], cmap="gray")
    ax.set_title(f"Gerçek: {y_test_sub[i]}\nTahmin: {pred_labels[i]}")
    ax.axis("off")
plt.suptitle("İlk 10 Test Tahmini")
plt.tight_layout()
plt.show()

def kağıttan_tahmin_et(dosya_yolu, egitilmis_model):
    yeni_img = cv2.imread(dosya_yolu, cv2.IMREAD_GRAYSCALE)
    if yeni_img is None:
        print("Hata: Dosya yolu geçersiz!")
        return

    img_resized = cv2.resize(yeni_img, (28, 28))
    img_inverted = cv2.bitwise_not(img_resized)
    islenmis_ozellikler = preprocess_image(img_inverted)
    hazir_veri = islenmis_ozellikler.reshape(1, 784)

    tahmin_olasiliklari = egitilmis_model.predict(hazir_veri)
    tahmin_edilen_rakam = np.argmax(tahmin_olasiliklari)
    eminlik_yuzdesi = np.max(tahmin_olasiliklari) * 100

    plt.imshow(img_inverted, cmap='gray')
    plt.title(f"Tahmin: {tahmin_edilen_rakam} | Eminlik: %{eminlik_yuzdesi:.2f}")
    plt.axis("off")
    plt.show()

# Kendi resim yolunuzu buraya girin:
# kağıttan_tahmin_et(r"resim_yolunuz.jpg", model)