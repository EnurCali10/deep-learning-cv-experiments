"""
GAN (Generative Adversarial Network) implementation in PyTorch.
fashion mnist veri seti ile moda urunu tasarimi yapacak bir model olusturulacak.

generator ve discriminator olmak uzere iki farkli neural network vardır 
generator aldıgı gürültülü veriyi görüntüye dönustürür ve discriminator gerçek ve sahte görüntüleri ayırt etmeye çalışır.

fashion mnist veri seti, 28x28 boyutunda gri tonlamalı görüntüler içerir ve 10 farklı sınıfa sahiptir (örneğin, tişört, pantolon, ayakkabı vb.).
 Bu veri seti, GAN'lerin moda ürünleri tasarlamak için kullanılabilir.
"""
import tensorflow as tf
from tensorflow.keras import layers 
import numpy as np
import matplotlib.pyplot as plt
import os 
from tensorflow.keras.datasets import fashion_mnist

BUFFER_SIZE = 60000
BATCH_SIZE = 128
NOISE_DIM = 100 #gürültü vektörünün boyutu, generatorun gürültülü veriyi görüntüye dönüştürmesi için kullanılır. Genellikle 100 boyutlu bir vektör tercih edilir.
IMG_SIZE=(28, 28, 1)#giris görüntü boyutu 
EPOCHS = 50

(train_images,_), (_,_) = fashion_mnist.load_data()#sadecee görüntüler etiketler kullanma
train_images = train_images.reshape(-1,28,28,1).astype('float32') 
train_images = (train_images - 127.5) / 127.5 #görüntüleri -1 ile 1 arasında normalize 
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(NOISE_DIM,))) #ilk tam bagli katman, gürültüyü ozellik haritasna dönüştürür. 7x7 boyutunda 256 filtre kullanır. use_bias=False, bias terimini kullanmaz.
    model.add(layers.BatchNormalization())#egitim stabilitesini artırır
    model.add(layers.LeakyReLU())#aktivasyon fonksiyonu olarak LeakyReLU kullanır. Bu, negatif değerler için küçük bir eğim sağlar ve böylece ölü nöron sorununu azaltır.
    model.add(layers.Reshape((7, 7, 256)))#tek boyutlu vektörü 3d ye cevir
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))  #son katman, 1 filtre kullanır (gri tonlamalı görüntü için) ve tanh aktivasyon fonksiyonu kullanır. Bu, çıktı değerlerini -1 ile 1 arasında sınırlar.
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

#generator = make_generator_model()

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1])) #ilk konvolüsyon katmanı, 64 filtre kullanır ve 5x5 boyutunda çekirdekler uygular. strides=(2, 2) ile görüntüyü yarıya indirir ve padding='same' ile kenarları korur.
    model.add(layers.LeakyReLU())#aktivasyon fonksiyonu olarak LeakyReLU kullanır. Bu, negatif değerler için küçük bir eğim sağlar ve böylece ölü nöron sorununu azaltır.
    model.add(layers.Dropout(0.3))#overfittingi onlemek icin dropout kullanir, %30 oranında nöronları rastgele kapatır.
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))#ikinci konvolüsyon katmanı, 128 filtre kullanır ve aynı şekilde 5x5 çekirdekler uygular.
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())#görüntüyü tek boyutlu vektöre dönüştürür.
    model.add(layers.Dense(1))#son katman, tek bir nöron içerir ve gerçek/sahte ayrımını yapmak için kullanılır. Aktivasyon fonksiyonu kullanılmaz çünkü daha sonra sigmoid fonksiyonu uygulanacaktır.
    return model 
   
#discriminator = make_discriminator_model()

#loss function 
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True) #binary cross-entropy loss fonksiyonu, gerçek ve sahte görüntüler arasındaki farkı ölçmek için kullanılır. from_logits=True, modelin çıktılarını doğrudan kullanır ve sigmoid fonksiyonunu uygulamaz.

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output) #gerçek görüntüler için loss hesaplanır. tf.ones_like(real_output) ile gerçek görüntüler için hedef değerler 1 olarak belirlenir.
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output) #sahte görüntüler için loss hesaplanır. tf.zeros_like(fake_output) ile sahte görüntüler için hedef değerler 0 olarak belirlenir.
    total_loss = real_loss + fake_loss #toplam loss, gerçek ve sahte görüntüler için hesaplanan lossların toplamıdır.
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output) #generatorun amacı, sahte görüntülerin gerçek gibi görünmesini sağlamaktır. Bu nedenle, sahte görüntüler için hedef değerler 1 olarak belirlenir.

generator=make_generator_model()
discriminator=make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-4) #Adam optimizasyon algoritması, generatorun ağırlıklarını güncellemek için kullanılır. Öğrenme hızı 0.0001 olarak belirlenmiştir.
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4) #Adam optimizasyon algoritması, discriminatorun ağırlıklarını güncellemek için kullanılır. Öğrenme hızı 0.0001 olarak belirlenmiştir.

#yardimci fonksiyon tanimla
seed = tf.random.normal([16, NOISE_DIM]) #gürültü vektörleri oluşturmak için kullanılır. 16 örnek ve NOISE_DIM boyutunda gürültü vektörleri oluşturulur.

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False) #modeli kullanarak test_input gürültü vektörlerinden görüntüler oluşturulur. training=False, modelin eğitim modunda olmadığını belirtir.
    fig = plt.figure(figsize=(4, 4)) #oluşturulan görüntüler 4x4 boyutunda bir figürde görselleştirilir.
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1) #her bir görüntü için subplot oluşturulur.
        plt.imshow((predictions[i, :, :, 0] +1)/2, cmap="gray")#görüntüler -1 ile 1 arasında normalize edildiği için, tekrar 0 ile 255 arasına dönüştürülür ve gri tonlamalı olarak görselleştirilir.
        plt.axis('off')

    if not os.path.exists('generated_images'):
        os.makedirs('generated_images') #oluşturulan görüntüler "generated_images" adlı bir klasöre kaydedilir. Eğer klasör mevcut değilse, os.makedirs() fonksiyonu ile oluşturulur.       

    plt.savefig(f"generated_images/image_at_epoch_{epoch:03d}.png") #oluşturulan figür kaydedilir.
    plt.close()  

#egitim fonksiyonu tanimla
def train(dataset, epochs):
    for epoch in range(1, epochs + 1):
        gen_loss_total = 0
        disc_loss_total = 0 
        batch_count = 0

        for image_batch in dataset:
            noise = tf.random.normal([BATCH_SIZE, NOISE_DIM]) #her bir batch için gürültü vektörleri oluşturulur.
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape: #gradient hesaplamak için iki ayrı GradientTape kullanılır. gen_tape generatorun ağırlıklarını güncellemek için, disc_tape ise discriminatorun ağırlıklarını güncellemek için kullanılır.
                generated_images = generator(noise, training=True) #generator modeli kullanılarak gürültü vektörlerinden sahte görüntüler oluşturulur.
                real_output = discriminator(image_batch, training=True) #discriminator modeli kullanılarak gerçek görüntüler değerlendirilir.
                fake_output = discriminator(generated_images, training=True) #discriminator modeli kullanılarak sahte görüntüler değerlendirilir.
                gen_loss = generator_loss(fake_output) #generatorun loss'u hesaplanır.
                disc_loss = discriminator_loss(real_output, fake_output) #discriminatorun loss'u hesaplanır.

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables) #generatorun ağırlıkları için gradyanlar hesaplanır.
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables) #discriminatorun ağırlıkları için gradyanlar hesaplanır.

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables)) #generatorun ağırlıkları güncellenir.
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables)) #discriminatorun ağırlıkları güncellenir.

            gen_loss_total += gen_loss
            disc_loss_total += disc_loss
            batch_count += 1
            print(f"Epoch {epoch}, Batch {batch_count}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}", end='\r')
            generate_and_save_images(generator, epoch, seed)

train(train_dataset, EPOCHS)            

            





