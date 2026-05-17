# EfficientNetB0 ile Pediatrik Kemik Yaşı Tahmini

## Proje Özeti
Bu depo, el radyografilerinden pediatrik kemik yaşını tahmin etmek amacıyla geliştirilmiş bir derin öğrenme regresyon modelini içermektedir. Model, RSNA Pediatric Bone Age Challenge veri seti kullanılarak eğitilmiştir.

Tahmin doğruluğunu artırmak için **çoklu modalite (multi-modal) girdi yaklaşımı** benimsenmiştir: Önceden eğitilmiş **EfficientNetB0** mimarisi kullanılarak X-ray görüntülerinden uzamsal öznitelikler çıkarılmış ve bu öznitelikler, regresyon çıktısı üreten tam bağlı (dense) katmanlara iletilmeden önce demografik veri (cinsiyet) ile birleştirilmiştir (concatenation).

## Model Mimarisi
* **Taban Model:** EfficientNetB0 (Ağırlıklar: ImageNet, İnce Ayar (Fine-tuning): Kapalı)
* **İkincil Girdi:** Cinsiyet (İkili/Float Formatında)
* **Çıktı Katmanı:** Regresyon (Ay cinsinden yaş) için doğrusal (linear) aktivasyonlu tek nöron.
* **Kayıp Fonksiyonu:** Ortalama Mutlak Hata (MAE)
* **Optimizasyon Algoritması:** Adam (Öğrenme Oranı: 0.001)
* **Geri Çağrılar (Callbacks):** EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

## Veri Seti
Çalışmada **RSNA Pediatric Bone Age Challenge** veri seti kullanılmıştır.
* **Kaynak:** [Kaggle - RSNA Pediatric Bone Age Challenge](https://www.kaggle.com/datasets/vaillant/rsna-pediatric-bone-age-challenge-n1200)
* **Hedef Değişken:** Ay cinsinden kemik yaşı (eğitim sırasında normalize edilmiştir).

## Gereksinimler
Kodu çalıştırmadan önce aşağıdaki bağımlılıkların yüklü olduğundan emin olun. İzole bir Python ortamı (örn. conda) kullanılması tavsiye edilir.

```bash
pip install numpy pandas matplotlib opencv-python scikit-learn tensorflow