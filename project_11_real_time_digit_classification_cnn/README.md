# Real-Time Handwritten Digit Classification with CNN

Bu projede, TensorFlow ve Keras kullanılarak Convolutional Neural Network (CNN) tabanlı bir el yazısı rakam sınıflandırma sistemi geliştirilmiştir.

Model, MNIST veri seti üzerinde eğitilmiş ve gerçek zamanlı görüntü işleme senaryolarında kullanılabilecek şekilde tasarlanmıştır.

---

## Proje Amacı

Bu çalışmanın amacı:

- CNN mimarisini kullanarak el yazısı rakamlarını sınıflandırmak,
- Veri artırma (Data Augmentation) teknikleri ile model performansını geliştirmek,
- Eğitilen modeli gerçek zamanlı kamera uygulamalarında kullanılabilecek hale getirmektir.

---

## Kullanılan Teknolojiler

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

---

## Veri Seti

Projede MNIST veri seti kullanılmıştır.

MNIST:
- 0-9 arasındaki rakamlardan oluşur
- 28x28 boyutunda gri tonlamalı görüntüler içerir
- El yazısı rakam sınıflandırma problemleri için standart benchmark veri setlerinden biridir

---

## Model Mimarisi

Model yapısı:

- Conv2D
- MaxPooling2D
- Conv2D
- MaxPooling2D
- Flatten
- Dense
- Softmax

katmanlarından oluşmaktadır.

---

## Veri Artırma (Data Augmentation)

Modelin genelleme performansını artırmak amacıyla:

- Döndürme (Rotation)
- Yakınlaştırma (Zoom)
- Yatay/Dikey kaydırma (Shift)

işlemleri uygulanmıştır.

---
