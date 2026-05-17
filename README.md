# Deep Learning Projects

Bu depo; temel yapay sinir ağlarından başlayarak, yüksek verimlilik odaklı nesne algılama, biyomekanik analiz ve üretken modellere (GAN) kadar uzanan geniş bir yelpazedeki makine öğrenimi ve bilgisayarlı görü projelerini içermektedir.

## Projeler

- **[project_1_ann](./project_1_ann/)** - Model: ANN | Veri Seti: MNIST | El yazısı rakam sınıflandırma
- **[project_2_cnn](./project_2_cnn/)** - Model: CNN | Veri Seti: TF Flowers | Çiçek türü sınıflandırma
- **[project_3_transfer_learning](./project_3_transfer_learning/)** - Model: DenseNet121 | Veri Seti: Chest X-Ray | Zatürre tespiti
- **[project_4_cifar10_resnet18](./project_4_cifar10_resnet18/)** - Model: ResNet-18 | Veri Seti: CIFAR-10 | Nesne tanıma ve hata analizi çalışması
- **[project_5_cifar100_resnet18](./project_5_cifar100_resnet18/)** - Model: ResNet-18 | Veri Seti: CIFAR-100 | Karmaşık veri setlerinde model kapasitesi ve hata analizi
- **[project_6_fashion_gan](./project_6_fashion_gan/)** - Model: DCGAN | Veri Seti: Fashion MNIST | Yapay zeka ile moda tasarımı ve ürün üretimi
- **[project_7_yolov8_traffic_signs](./project_7_yolov8_traffic_signs/)** - Model: YOLOv8 | Veri Seti: Roboflow Traffic Signs | Gerçek zamanlı trafik işareti algılama ve sınıflandırma
- **[project_8_people_counting](./project_8_people_counting/)** - Model: YOLOv8n | Veri Seti: Kaggle (CrowdUIT) | Dikey çizgi geçişine dayalı çift yönlü insan sayma sistemi
- **[project_9_vehicle_tracking](./project_9_vehicle_tracking/)** - Model: YOLOv8m + ByteTrack | Veri Seti: Custom Video | Gerçek zamanlı araç tespiti, sınıflandırma ve nesne takibi
- **[project_10_neural_style_transfer_vgg19](./project_10_neural_style_transfer_vgg19/)** - Model: VGG19 | Veri Seti: Custom (İçerik ve Stil Görselleri) | Önceden eğitilmiş evrişimli ağların (CNN) derin özellik uzaylarında (feature space) sanatsal stil ve içerik optimizasyonu
- **[project_11_real_time_digit_classification_cnn](./project_11_real_time_digit_classification_cnn/)** - Model: CNN | Veri Seti: Custom (Webcam) | Gerçek zamanlı görüntü akışı üzerinden el yazısı rakam tanıma ve sınıflandırma
- **[project_12_pediatric_bone_age_efficientnet](./project_12_pediatric_bone_age_efficientnet/)** - Model: EfficientNetB0 | Veri Seti: RSNA Bone Age | Çoklu modalite (radyografi + cinsiyet) kullanılarak pediatrik kemik yaşı regresyonu
- **[project_13_kinematic_squat_analysis](./project_13_kinematic_squat_analysis/)** - Model: MediaPipe Pose | Veri Seti: Özel Video | Eklem koordinatları üzerinden biyomekanik açı hesaplama ve durum makinesi (state machine) ile analiz
- **[project_14_aerial_segmentation_unet](./project_14_aerial_segmentation_unet/)** - Model: U-Net | Veri Seti: Aerial Imagery | Havadan çekilmiş yüksek çözünürlüklü uydu görüntüleri üzerinde anlamsal (semantic) segmentasyon
- **[project_15_heuristic_facial_emotion_tracker](./project_15_heuristic_facial_emotion_tracker/)** - Model: MediaPipe Face Mesh | Veri Seti: Gerçek Zamanlı | Yüz eklem noktaları (landmarks) arasındaki geometrik oranlara dayalı heuristik duygu tahmini
- **[project_16_facial_emotion_recognition_cnn](./project_16_facial_emotion_recognition_cnn/)** - Model: Özel CNN | Veri Seti: FER2013 | Gürültüye dayanıklı, toplu normalizasyon (batch normalization) ve veri artırımlı (data augmentation) yüz ifadesi sınıflandırması

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
