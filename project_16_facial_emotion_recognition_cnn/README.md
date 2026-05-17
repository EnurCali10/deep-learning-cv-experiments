# Facial Emotion Recognition via Convolutional Neural Networks (FER2013)

## Proje Özeti
Bu proje, kontrolsüz ortamlarda çekilmiş düşük çözünürlüklü (48x48) yüz görüntülerinden 7 temel duygu durumunu (Mutluluk, Üzüntü, Korku, Öfke, Şaşkınlık, İğrenme, Nötr) sınıflandırmak için tasarlanmış derin bir Evrişimli Sinir Ağı (CNN) mimarisidir. Çalışma, FER2013 veri seti üzerinde gerçekleştirilmiş olup, gürültüye karşı dayanıklılığı (robustness) artırmak amacıyla çeşitli optimizasyon tekniklerini barındırmaktadır.

## Model Mimarisi ve Optimizasyon Stratejileri
Sıfırdan (from scratch) eğitilen bu CNN modeli, aşırı öğrenmeyi (overfitting) engellemek ve yakınsama (convergence) hızını artırmak için şu mimari kararları içerir:
* **Veri Artırımı (Data Augmentation):** Küçük ve dengesiz veri setini dengelemek için rotasyon, kaydırma (shift), yakınlaştırma (zoom) ve yatay çevirme (flip) işlemleri uygulanmıştır.
* **Toplu Normalizasyon (Batch Normalization):** Her evrişim bloğunda aktivasyon dağılımlarını normalize ederek içsel ortak değişken kaymasını (internal covariate shift) azaltır.
* **Dropout (%50):** Tam bağlı (Dense) katmanda nöronların yarısı rastgele kapatılarak ağın ezberleme yapması engellenmiştir.
* **Dinamik Öğrenme Oranı (ReduceLROnPlateau):** Validasyon kaybı (val_loss) durağanlaştığında öğrenme oranı logaritmik olarak düşürülür.

## Veri Seti
* **FER2013:** 28.709 eğitim, 7.178 test görseli (Gri tonlamalı, 48x48 piksel).
* **Kaynak:** [Kaggle - FER2013](https://www.kaggle.com/datasets/msambare/fer2013)

## Kurulum ve Gereksinimler
İzole bir Python ortamı kullanılması önerilir:
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn