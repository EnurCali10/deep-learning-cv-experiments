# Project 6: Fashion Product Synthesis with DCGANs

Bu proje, **Fashion MNIST** veri setini kullanarak gerçekçi moda ürünleri (tişört, ayakkabı, çanta vb.) tasarlayabilen bir Derin Konvolüsyonel Üretken Çekişmeli Ağ (DCGAN) uygulamasıdır.

##  Mimari Yapı
Proje, birbiriyle sürekli bir rekabet (minimax oyunu) içerisinde olan iki derin ağdan oluşur:

### Generator (Üretici)
- **Girdi:** 100 boyutlu Gauss gürültüsü (Noise).
- **İşlem:** `Conv2DTranspose` katmanlarını kullanarak 7x7 boyutundaki özellikleri 28x28 boyutunda bir moda görseline dönüştürür.
- **Aktivasyon:** Çıkış katmanında `tanh` kullanılarak pikseller [-1, 1] aralığına normalize edilmiştir.

### Discriminator (Ayırt Edici)
- **Girdi:** 28x28 boyutunda görsel (Gerçek veya Üretilmiş).
- **İşlem:** Evrişimli katmanlar (Conv2D) ve `LeakyReLU` aktivasyonu ile görselin orijinalliğini denetler.
- **Düzenlileştirme:** `Dropout(0.3)` kullanılarak aşırı öğrenme (overfitting) engellenmiştir.



##  Eğitim Detayları ve Kayıp Fonksiyonu
- **Loss:** Binary Cross-Entropy (from_logits=True).
- **Optimizer:** Adam (Learning Rate: 1e-4).
- **Epoch:** 50.
- **Gözlem:** Eğitim ilerledikçe üreticinin (Generator) ürettiği "bulanık" silüetlerin, ayırt edicinin (Discriminator) geri bildirimleri sayesinde net moda ürünlerine dönüştüğü gözlemlenmiştir.

##  Green AI 
DCGAN modellerinin eğitimi ciddi GPU kaynakları tüketebilir. Bu projede:
- **RTX 3050** donanımı üzerinde batch bazlı optimizasyon yapılarak eğitim süresi kısaltılmıştır.
- Verimli `tf.data` API'si kullanılarak CPU-GPU arası veri aktarımı minimize edilmiş ve enerji tasarrufu sağlanmıştır.

##  Kurulum ve Kullanım
Proje için gerekli kütüphaneler: `tensorflow`, `numpy`, `matplotlib`.

```bash
python fashion_gan.py