"""
problem tanimi: gerçek zamanlı goruntu isleme ile rakamlari sinifladirma
mnist veri seti ile cnn egitimi gerceklestirme, bu modeli kaydetme 
kamera ile birlikte kagitlara yazmis oldugumuz rakamalari siniflandirmaya calisma
mnist: rakamalardan olusan bir veri seti, 0-9 arasinda rakamlari iceren 28x28 boyutunda goruntuleri barindirir
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist 
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# mnist veri setini yukleme
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train= 255-x_train
x_test= 255-x_test

plt.figure(figsize=(10, 10))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# veriyi normalleştirme
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# data augmentation (veri arttırma)
datagen = ImageDataGenerator(
    rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1
)

# CNN modelini tanimlama
model = keras.Sequential(
    [
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

print(model.summary())

# modeli derleme
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# modeli egitme
model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=5, validation_data=(x_test, y_test))

# modeli kaydetme
model.save("mnist_cnn_model.h5")

# modeli degerlendirme
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
