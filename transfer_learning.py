"""
transfer learning : önceden çok büyük veri setleri üzerinde eğitilmiş bir modelin, yeni bir görev için kullanılmasıdır. Bu yöntem, özellikle sınırlı veri setlerine sahip durumlarda, modelin genel özellikleri öğrenmiş olması nedeniyle daha iyi performans sağlar.
zatüre sınıflandırma için transfer learning uygulaması
transfer learning model: densenet121
"""
#import libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator#goruntu verisi yükleme ve data augmention için
from tensorflow.keras.applications import DenseNet121#transfer learning modeli
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, ReduceLROnPlateau#callbackler
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Dropout #modelin sonuna eklenecek katmanlar
from tensorflow.keras.optimizers import Adam#modelin derlenmesi için optimizer
from tensorflow.keras.models import Model#model oluşturmak için


import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import  confusion_matrix, ConfusionMatrixDisplay #karisiklik matrisi ve görselleştirme

#data augmention
train_datagen = ImageDataGenerator(
    rescale=1./255, #görüntüleri normalize etmek için piksel değerlerini 0-1 aralığına dönüştürür
    horizontal_flip=True, #görüntüleri yatay olarak rastgele çevirir
    rotation_range=10, #görüntüleri rastgele döndürür
    brightness_range=[0.8, 1.2], #görüntülerin parlaklığını rastgele değiştirir
    validation_split=0.1 #valiation seti için veri setinin %10'unu ayırır

) #sınıfı, görüntü verisi üzerinde çeşitli dönüşümler yaparak veri setini genişletmek için kullanılır. Bu, modelin daha fazla çeşitlilikte veri görmesini sağlar ve overfitting'i azaltabilir. Örneğin, döndürme, kaydırma, zoom yapma gibi işlemler uygulanabilir.
#train data= train + validation
test_datagen = ImageDataGenerator(rescale=1./255) #test verisi için sadece normalizasyon yapılır

DATA_DIR = r"PATH/TO/YOUR/chest_xray"  # Kendi veri seti yolunuzu girin
IMG_SIZE = (224, 224) #modelin girdi boyutu
BATCHSIZE = 64
CLASS_MODE="binary" #binary sınıflandırma


train_gen = train_datagen.flow_from_directory(
     os.path.join(DATA_DIR, "train"), 
     target_size=IMG_SIZE,#modelin girdi boyutuna uygun hale getirmek için görüntüleri yeniden boyutlandırır
     batch_size=BATCHSIZE, 
     class_mode=CLASS_MODE,
     subset="training", #train seti için veri setinin %90'ını kullanır
     shuffle=True #veri setini karıştırarak modelin farklı örnekler görmesini sağlar
)

val_gen = train_datagen.flow_from_directory(
     os.path.join(DATA_DIR, "train"),
     target_size=IMG_SIZE,
     batch_size=BATCHSIZE,
     class_mode=CLASS_MODE,
     subset="validation", #validation seti için veri setinin %10'unu kullanır
     shuffle=False #validation seti için veri setini karıştırmaz, böylece modelin performansını daha doğru değerlendirebiliriz
)

test_gen = test_datagen.flow_from_directory(
     os.path.join(DATA_DIR, "test"),
     target_size=IMG_SIZE,
     batch_size=BATCHSIZE,
     class_mode=CLASS_MODE,
     shuffle=False
)


#basic visualization
class_names = list(train_gen.class_indices.keys()) #sınıf isimlerini alır[normal,zatüre]
images, labels = next(train_gen) #train generatoründen bir batch görüntü ve etiket alır

def plot_images(images, labels, class_names):
    plt.figure(figsize=(10, 4))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[int(labels[i])])
        plt.axis("off")
    plt.tight_layout()    
    plt.show()


#transfer learning modelin tanimlanmasi: densenet121
base_model = DenseNet121(
    weights="imagenet", #önceden eğitilmiş ağırlıkları kullanır
    include_top=False, #modelin sonundaki sınıflandırma katmanını dahil etmez
    input_shape=(*IMG_SIZE, 3) #modelin girdi boyutu
)
base_model.trainable = False #base modelin ağırlıklarını dondurur, böylece eğitim sırasında güncellenmezler

x = base_model.output #base modelin çıktısını alır
x=GlobalAveragePooling2D()(x)
x=Dense(128, activation="relu")(x) #bir fully connected katman ekler
x=Dropout(0.5)(x) #overfitting'i azaltmak için dropout katmanı ekler
pred=Dense(1, activation="sigmoid")(x) #binary sınıflandırma için sigmoid aktivasyonlu bir çıkış katmanı ekler

model= Model(inputs = base_model.input, outputs=pred) #base modelin girişini ve yeni eklenen katmanların çıkışını birleştirerek yeni bir model oluşturur

#modelin derlenmesi ve callback ayarlari

model.compile(
    optimizer=Adam(learning_rate=1e-4), #modelin ağırlıklarını güncellemek için Adam optimizasyon algoritmasını kullanır
    loss="binary_crossentropy", #binary sınıflandırma için kayıp fonksiyonu olarak binary crossentropy kullanır
    metrics=["accuracy"] #modelin performansını değerlendirmek için doğruluk metriğini kullanır
)

callbacks =[
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True), #validation kaybını izleyerek erken durdurma uygular
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6), #validation kaybı iyileşmediğinde öğrenme oranını azaltır
    ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True) #en iyi modeli kaydeder
]

print(model.summary())

#modelin egitimleri ve sonuclarin degerlendirilmesi
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=2,
    callbacks=callbacks,
    verbose=1 #eğitim sırasında ilerleme çubuğu ve eğitim süresi gibi bilgileri gösterir
)

pred_probs= model.predict(test_gen, verbose=1) #test verisi üzerinde tahmin yapar
pred_labels = (pred_probs > 0.5).astype(int).astype(int).ravel() #tahmin edilen olasılıkları sınıf etiketlerine dönüştürür (0 veya 1)
true_labels = test_gen.classes #test verisinin gerçek sınıf etiketlerini alır
cm = confusion_matrix(true_labels, pred_labels) #karışıklık matrisini hesaplar
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names) #karışıklık matrisini görselleştirir

plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues) #karışıklık matrisini mavi tonlarda görselleştirir
plt.title("Test Seti Confusion Matrix")
plt.show()
