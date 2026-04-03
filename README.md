# Deep Learning Projects

Bu repo; temel yapay sinir ağlarından başlayarak, yüksek verimlilik odaklı nesne algılama, takip ve üretken modellere (GAN) kadar uzanan geniş bir yelpazedeki derin öğrenme projelerini içermektedir.

## Projeler

- **[project_1_ann](./project_1_ann/)** - Model: ANN | Veri Seti: MNIST | El yazısı rakam sınıflandırma
- **[project_2_cnn](./project_2_cnn/)** - Model: CNN | Veri Seti: TF Flowers | Çiçek türü sınıflandırma
- **[project_3_transfer_learning](./project_3_transfer_learning/)** - Model: DenseNet121 | Veri Seti: Chest X-Ray | Zatürre tespiti
- **[project_4_cifar10_resnet18](./project_4_cifar10_resnet18/)** - Model: ResNet-18 | Veri Seti: CIFAR-10 | Nesne tanıma ve hata analizi çalışması
- **[project_5_cifar100_resnet18](./project_5_cifar100_resnet18/)** - Model: ResNet-18 | Veri Seti: CIFAR-100 | Karmaşık veri setlerinde model kapasitesi ve hata analizi
- **[project_6_fashion_gan](./project_6_fashion_gan/)** - Model: DCGAN | Veri Seti: Fashion MNIST | Yapay zeka ile moda tasarımı ve ürün üretimi
- **[project_7_yolov8_traffic_signs](./project_7_yolov8_traffic_signs/)** - Model: YOLOv8 | Veri Seti: Roboflow Traffic Signs | Gerçek zamanlı trafik işareti algılama ve sınıflandırma
- **[project_8_vehicle_tracking](./project_8_vehicle_tracking/)** - Model: YOLOv8m + ByteTrack | Veri Seti: Custom Video | Gerçek zamanlı araç tespiti, sınıflandırma ve nesne takibi
- **[project_9_people_counting](./project_9_people_counting/)** - Model: YOLOv8n | Veri Seti: Kaggle (CrowdUIT) | Dikey çizgi geçişine dayalı çift yönlü insan sayma sistemi
- **[project_10_vehicle_counting](./project_10_vehicle_counting/)** - Model: YOLOv8n | Veri Seti: Kaggle (Car Detection) | Çapraz çizgi algoritması ile araç türü bazlı trafik sayımı



## Gereksinimler

```bash
pip install tensorflow ultralytics opencv-python matplotlib numpy scikit-learn tensorflow-datasets torch torchvision
```

## Kurulum

```bash
git clone https://github.com/EnurCali10/dl_projects.git
cd dl_projects
pip install -r requirements.txt
```
