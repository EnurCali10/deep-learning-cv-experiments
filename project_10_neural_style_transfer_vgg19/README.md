# VGG19 ile Neural Style Transfer

Bu projede, PyTorch kullanılarak önceden eğitilmiş (pretrained) VGG19 modeli ile Neural Style Transfer uygulaması geliştirilmiştir.

Amaç, bir görselin içerik bilgisini korurken başka bir görselin sanatsal stilini hedef görüntü üzerine aktarmaktır.

---

## Proje Özellikleri

- Pretrained VGG19 modeli kullanımı
- Gram Matrix tabanlı stil çıkarımı
- İçerik ve stil kaybı (loss) optimizasyonu
- GPU destekli çalışma
- PyTorch tabanlı implementasyon

---

## Kullanılan Teknolojiler

- Python
- PyTorch
- Torchvision
- Matplotlib
- PIL

---

## Çalışma Mantığı

Model:

- İçerik görselinden içerik özelliklerini,
- Stil görselinden ise stil özelliklerini

VGG19 ağının ara katmanları üzerinden çıkarmaktadır.

Stil benzerliği Gram Matrix yöntemi ile hesaplanırken, içerik bilgisi özellik haritaları arasındaki fark kullanılarak korunmaktadır.

Hedef görüntü, optimizasyon süreci boyunca güncellenerek yeni bir stilize edilmiş görüntü oluşturulmaktadır.

---

