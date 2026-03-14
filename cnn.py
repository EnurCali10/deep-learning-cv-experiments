"""
flowers dataset:
rgb:224x224
cnn ile sınıflandırma modeli olusturma ve problemi çözme

"""
#import libraries
from tensorflow_datasets import load #veriseti yükleme
from tensorflow.data import AUTOTUNE #veri seti işlemlerini hızlandırmak için
from tensorflow.keras.models import Sequential #model oluşturmak için
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout #cok boyutlu veriyi tek boyutlu hale getirme , #model katmanları,#dropout: overfitting'i önlemek için
from tensorflow.keras.optimizers import Adam #optimizasyon algoritması
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint #callbacks
import tensorflow as tf
import matplotlib.pyplot as plt

#veri seti yükleme
(ds_train, ds_val), ds_info = load(
    'tf_flowers', 
    split=['train[:80%]', #veriseti %80 eğitim, %20 test olarak bölünür
           'train[80%:]'],
    as_supervised=True, #veri seti etiketli olarak yüklenir 
    with_info=True #veri seti hakkında bilgi verir     
)
print(ds_info.features)#veri seti özelliklerini yazdırır
print("Sınıf sayısı:", ds_info.features['label'].num_classes) #sınıf sayısını yazdırır

#ornek veri gorsellestirme
#egitim setinden rastgele 3 örnek alır ve görselleştirir
fig = plt.figure(figsize=(10, 5))
for i, (image, label) in enumerate(ds_train.take(3)):
    ax = fig.add_subplot(1, 3, i + 1)#1 satir 3 sutun ve i+1. resimi ekler
    ax.imshow(image.numpy().astype("uint8"))#resmi gorselleştirir
    ax.set_title(f"Label: {label.numpy()}")#etiketi başlık olarak ekler
    ax.axis("off")#ekseni kapatır
plt.tight_layout()#görselleştirme düzenini sıkıştırır
plt.show()

IMG_SIZE = (180,180) #resim boyutu

#data augmentation + preprocessing
def preprocess_train(image, label):
    """
    resize,random flip, brightness, contrast, crop, normalize
    """
    image = tf.image.resize(image, IMG_SIZE) #resmi belirtilen boyuta yeniden boyutlandırır
    image = tf.image.random_flip_left_right(image) #resmi yatay olarak rastgele çevirir
    image = tf.image.random_brightness(image, max_delta=0.1) #resmin parlaklığını rastgele değiştirir
    image = tf.image.random_contrast(image, lower=0.9, upper=1.2) #resmin kontrastını rastgele değiştirir
    image = tf.image.random_crop(image, size=(160,160,3)) # resmi rastgele kırparak belirtilen boyuta getirir
    image = tf.image.resize(image, IMG_SIZE) #resmi belirtilen boyuta yeniden boyutlandırır
    image = tf.cast(image, tf.float32) / 255.0 #resmi normalize eder
    return image, label

def preprocess_val(image, label):
    """
    resize, normalize
    """
    image = tf.image.resize(image, IMG_SIZE) #resmi belirtilen boyuta yeniden boyutlandırır
    image = tf.cast(image, tf.float32) / 255.0 #resmi   normalize eder
    return image, label         

#veriseti hazırlama
ds_train = (
    ds_train
     .map(preprocess_train, num_parallel_calls=AUTOTUNE) #egitim veriseti için ön işleme fonksiyonunu uygular
     .shuffle(1000) #veriseti karıştırır
     .batch(32) #veriseti 32'lik gruplara bölün     
     .prefetch(AUTOTUNE) #veriseti işlemlerini hızlandırmak için önceden yükleme yapar 
) 


ds_val = (
    ds_val
    .map(preprocess_val, num_parallel_calls=AUTOTUNE) #doğrulama veriseti için ön işleme fonksiyonunu uygular
    .batch(32) #veriseti 32'lik gruplara bölün 
    .prefetch(AUTOTUNE) #veriseti işlemlerini hızlandırmak için önceden yükleme yapar
)

#cnn modeli olusturma
model=Sequential([
    Input(shape=(180,180, 3)), #giriş katmanı, resim boyutu ve renk kanalları
    #FEATURE EXTRACTION
    Conv2D(32,(3,3), activation='relu', input_shape=(180,180, 3)), #ilk konvolüsyon katmanı, 32 filtre, 3x3 kernel boyutu, ReLU aktivasyon fonksiyonu
    MaxPooling2D((2,2)), #ilk max pooling katmanı, 2x2 havuzlama boyutu

    Conv2D(64,(3,3), activation='relu'), #ikinci konvolüsyon katmanı, 64 filtre, 3x3 kernel boyutu, ReLU aktivasyon fonksiyonu  
    MaxPooling2D((2,2)), #ikinci max pooling katmanı, 2x2 havuzlama boyutu

    Conv2D(128,(3,3), activation='relu'), #üçüncü konvolüsyon katmanı, 128 filtre, 3x3 kernel boyutu, ReLU aktivasyon fonksiyonu
    MaxPooling2D((2,2)), #üçüncü max pooling katmanı
    
    #CLASSIFICATION
    Flatten(), #çok boyutlu veriyi tek boyutlu hale getirir 
    Dense(128, activation='relu'), #tam bağlantılı katman, 128 nöron, ReLU aktivasyon fonksiyonu
    Dropout(0.5), #overfitting'i önlemek için dropout katmanı       
    Dense(ds_info.features['label'].num_classes, activation='softmax') #çıkış katmanı, sınıf sayısı kadar nöron, softmax aktivasyon fonksiyonu

])
"""
kernel 3x3 matris küçük olması komşu pikseller arasındaki ilişkileri yakalamaya yardımcı olur, parametre sayısı azaltır ve modelin daha hızlı öğrenmesini sağlar.
ağ derinleştikçe daha karmaşık özellikler öğrenebilir, ancak aşırı derin ağlar overfitting'e neden olabilir, bu nedenle denge önemlidir.
max pooling, özellik haritalarını küçültür ve önemli özellikleri korur, böylece modelin genelleme yeteneği artar ve hesaplama maliyeti azalır.Modelin toplam parametre sayısını makul tutarken modelin öğrenme kapasitesini artırır.

"""

#callback

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True), #eger validation loss 3 epoch boyunca iyileşmezse eğitimi durdur en iyi agırlıkları yükle
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, min_lr=1e-9), #eger validation loss 3 epoch boyunca iyileşmezse öğrenme oranını 0.2 ile çarpanı ile azalt, verbose=1 ile öğrenme oranı değiştiğinde bilgi verir, min_lr ile öğrenme oranının minimum değerini belirler
model_checkpoint= ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True), #validation loss en iyi olduğunda modeli 'best_model.h5' olarak kaydet

my_callbacks = [early_stopping, reduce_lr, model_checkpoint]
#derleme 
model.compile(
    optimizer=Adam(learning_rate=0.001), #Adam optimizasyon algoritması, öğrenme oranı 0.001
    loss='sparse_categorical_crossentropy', #kayıp fonksiyonu, çok sınıflı sınıflandırma için uygun
    metrics=['accuracy'] #değerlendirme metriği, doğruluk
)

print(model.summary())

#modeli egitme
history = model.fit(
    ds_train, #egitim veriseti
    validation_data=ds_val, #doğrulama veriseti
    epochs=30, #epoch sayısı
    callbacks=my_callbacks ,#callbacks
    verbose=1 #eğitim sürecini ayrıntılı olarak gösterir
)

#modelin degerlendirmesi
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu') #egitim doğruluğunu çiz
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu') #doğrulama doğruluğunu çiz
plt.title('Doğruluk Grafiği') 
plt.xlabel('Epoch')
plt.ylabel('Doğruluk') 
plt.legend()

#loss grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı') #egitim kaybını çiz
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı') #doğrulama kaybını çiz
plt.title('Kayıp Grafiği') 
plt.xlabel('Epoch') 
plt.ylabel('Kayıp')
plt.legend() 
plt.tight_layout() 
plt.show() 
