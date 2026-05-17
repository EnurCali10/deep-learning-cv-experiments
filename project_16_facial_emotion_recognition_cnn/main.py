"""
problem tanimi:Bu çalışma, 28.709 eğitim ve 7.178 test görselinden oluşan FER2013 veri setini kullanarak,
 48x48 piksel boyutundaki gri tonlamalı yüz görüntülerinden yedi farklı temel duyguyu (mutluluk, üzüntü, korku, öfke, şaşkınlık, iğrenme, nötr) sınıflandırabilen derin bir evrişimli sinir ağı (CNN) mimarisinin geliştirilmesini ve 
 optimize edilmesini amaçlamaktadır. Proje kapsamında temel problem; düşük çözünürlüklü ve kontrolsüz ortamlarda çekilmiş yüz ifadelerinden, yüksek doğruluk oranı ve düşük kayıp değerleri ile anlamsal öznitelik çıkarımı yapabilecek gürültüye dayanıklı bir modelin tasarlanmasıdır.

dataset:https://www.kaggle.com/datasets/msambare/fer2013

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import ( Conv2D, MaxPooling2D, Dense, Dropout,Flatten, BatchNormalization, Input)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


#veriseti path
train_dir = "dataset/train"
test_dir  = "dataset/test"

#parametreler
IMG_SIZE   = 48
BATCH_SIZE = 64
EPOCHS     = 30

#  DATA AUGMENTATION 
#  FER2013 küçük ve dengesiz bir veri seti.
# Aynı görseli tekrar tekrar görmek modeli ezberlemeye iter (overfitting).
# Augmentation, gerçek dünya varyasayımlarını simüle eder:
#   - rotation_range   : Hafif açısal bozulma (yüzler her zaman tam dik değil)
#   - zoom_range        : Yakın / uzak çekim simülasyonu
#   - width/height shift: Yüzün karedeki konumunu değiştirme
#   - horizontal_flip   : Ayna yansıması (duygu ifadeleri simetriktir → güvenli)
#   - rescale           : [0,255] → [0,1] normalizasyon, gradient stabilitesi

train_datagen = ImageDataGenerator(
    rescale           = 1./255,
    rotation_range    = 20,
    zoom_range        = 0.2,
    width_shift_range = 0.2,
    height_shift_range= 0.2,
    horizontal_flip   = True,
    validation_split  = 0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Generatorlar
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size  = (IMG_SIZE, IMG_SIZE),
    batch_size   = BATCH_SIZE,
    color_mode   = 'grayscale',
    class_mode   = 'categorical',
    subset       = 'training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size  = (IMG_SIZE, IMG_SIZE),
    batch_size   = BATCH_SIZE,
    color_mode   = 'grayscale',
    class_mode   = 'categorical',
    subset       = 'validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size  = (IMG_SIZE, IMG_SIZE),
    batch_size   = BATCH_SIZE,
    color_mode   = 'grayscale',
    class_mode   = 'categorical',
    shuffle      = False
)

class_labels = list(train_generator.class_indices.keys())
print("\n🎭 Sınıflar:", class_labels)


# GÖRSEL: Augmentation Mantığı — Aynı Görselın 8 Farklı Hali
def goster_augmentation_ornekleri(generator, n=8):
    """
    Eğitim generatöründen bir batch alır,
    aynı sınıftan n örnek yan yana göstererek
    augmentation'ın ne yaptığını açıklar.
    """
    batch_imgs, batch_labels = next(generator)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.suptitle(
        "Veri Augmentation — Aynı Sınıftan Üretilen Farklı Örnekler\n"
        "(Döndürme · Zoom · Kaydırma · Yatay Çevirme)",
        fontsize=13, fontweight='bold'
    )

    for i, ax in enumerate(axes.flat):
        if i >= len(batch_imgs):
            ax.axis('off')
            continue
        ax.imshow(batch_imgs[i].squeeze(), cmap='gray')
        sinif_idx = np.argmax(batch_labels[i])
        ax.set_title(f"Sınıf: {class_labels[sinif_idx]}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("augmentation_ornekleri.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Kaydedildi: augmentation_ornekleri.png")

goster_augmentation_ornekleri(train_generator)


#CNN MODEL

model = Sequential(name="FER2013_CNN")

# CONV BLOK 1  →  32 filtre, 3×3
# Conv2D  : 32 küçük filtre görüntü üzerinde kayar; kenar/köşe gibi
#           düşük seviye özellikler öğrenilir.
# BatchNorm: Aktivasyon dağılımını μ=0, σ=1'e normalize eder.
#            → Daha hızlı yakınsama, vanishing gradient azalır.
# MaxPool : 2×2 pencereyle boyutu yarıya indirir (48→24).
#           Öteleme değişmezliği (translation invariance) kazanılır.
model.add(Conv2D(32, (3,3), activation='relu',
                 input_shape=(IMG_SIZE, IMG_SIZE, 1),
                 padding='same', name='conv1'))
model.add(BatchNormalization(name='bn1'))
model.add(MaxPooling2D(pool_size=(2,2), name='pool1'))

# CONV BLOK 2  →  64 filtre, 3×3
# Filtre sayısı 2 katına çıkar (32→64); daha fazla özellik kanalı.
# Orta seviye: göz çevresi, burun yapısı, ağız köşeleri.
model.add(Conv2D(64, (3,3), activation='relu',
                 padding='same', name='conv2'))
model.add(BatchNormalization(name='bn2'))
model.add(MaxPooling2D(pool_size=(2,2), name='pool2'))

# CONV BLOK 3  →  128 filtre, 3×3
# Yüksek seviye: kaş çatma, gülümseme, şaşkınlık gibi tam ifadeler.
model.add(Conv2D(128, (3,3), activation='relu',
                 padding='same', name='conv3'))
model.add(BatchNormalization(name='bn3'))
model.add(MaxPooling2D(pool_size=(2,2), name='pool3'))

# FLATTEN  →  3D haritayı 1D vektöre dönüştür
model.add(Flatten(name='flatten'))

# DENSE  →  256 nöron, ReLU
model.add(Dense(256, activation='relu', name='dense1'))

# ──────────────────────────────────────────────────────────────
# DROPOUT  →  p = 0.5
# ──────────────────────────────────────────────────────────────
# Eğitimde her adımda nöronların %50'si rastgele kapatılır.
# → Nöronlar birbirine bağımlı hale gelemez (co-adaptation önlenir).
# → 2^n farklı ağ kombinasyonu → ensemble learning etkisi.
# Test sırasında tüm nöronlar aktif; ağırlıklar p ile ölçeklenir.
model.add(Dropout(0.5, name='dropout1'))


# OUTPUT  →  7 sınıf, Softmax
model.add(Dense(len(class_labels), activation='softmax', name='output'))

print("\n" + "="*60)
print(" MODEL MİMARİSİ")
print("="*60)
model.summary()


# GÖRSEL: CNN Katman Açıklama Diyagramı

def goster_katman_aciklamalari(model):
    """
    Her katmanın adını, çıktı boyutunu ve parametre sayısını
    renkli bir çubuk diyagramı olarak gösterir.
    """
    katman_adlari   = []
    cikti_boyutlari = []
    parametre_sayilari = []

    for layer in model.layers:
        katman_adlari.append(layer.name)
        try:
            cikti_boyutlari.append(str(layer.output_shape))
        except Exception:
            cikti_boyutlari.append("?")
        parametre_sayilari.append(layer.count_params())

    renkler = []
    for name in katman_adlari:
        if   'conv'    in name: renkler.append('#4A90D9')
        elif 'bn'      in name: renkler.append('#F5A623')
        elif 'pool'    in name: renkler.append('#7ED321')
        elif 'flatten' in name: renkler.append('#9B59B6')
        elif 'dense'   in name: renkler.append('#E74C3C')
        elif 'dropout' in name: renkler.append('#95A5A6')
        elif 'output'  in name: renkler.append('#1ABC9C')
        else:                   renkler.append('#BDC3C7')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("CNN Katman Açıklamaları", fontsize=14, fontweight='bold')

    # Sol: parametre sayısı
    bars = ax1.barh(katman_adlari, parametre_sayilari, color=renkler, edgecolor='white')
    ax1.set_xlabel("Parametre Sayısı")
    ax1.set_title("Katman Başına Parametre Sayısı")
    ax1.invert_yaxis()
    for bar, val in zip(bars, parametre_sayilari):
        if val > 0:
            ax1.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                     f'{val:,}', va='center', fontsize=8)

    # Sağ: açıklama tablosu
    ax2.axis('off')
    tablo_verisi = list(zip(katman_adlari, cikti_boyutlari,
                             [f"{p:,}" for p in parametre_sayilari]))
    tablo = ax2.table(
        cellText   = tablo_verisi,
        colLabels  = ['Katman', 'Çıktı Boyutu', 'Parametreler'],
        loc        = 'center',
        cellLoc    = 'left'
    )
    tablo.auto_set_font_size(False)
    tablo.set_fontsize(9)
    tablo.scale(1, 1.6)

    # Renk legend
    legend_items = [
        ('#4A90D9', 'Conv2D'),
        ('#F5A623', 'BatchNorm'),
        ('#7ED321', 'MaxPooling'),
        ('#9B59B6', 'Flatten'),
        ('#E74C3C', 'Dense'),
        ('#95A5A6', 'Dropout'),
        ('#1ABC9C', 'Output'),
    ]
    handles = [plt.Rectangle((0,0),1,1, color=c) for c, _ in legend_items]
    ax2.legend(handles, [l for _, l in legend_items],
               loc='lower center', ncol=4, fontsize=8,
               title='Katman Türü Renk Kodu')

    plt.tight_layout()
    plt.savefig("katman_aciklamalari.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(" Kaydedildi: katman_aciklamalari.png")

goster_katman_aciklamalari(model)

# GÖRSEL: BatchNorm + Dropout Açıklaması
def goster_batchnorm_dropout_aciklamasi():
    """
    BatchNorm: aktivasyon dağılımını normalize etmenin etkisini
    Dropout  : nöron devre-dışı bırakma örüntüsünü gösterir.
    """
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle("BatchNorm + Dropout — Neden Kullanıyoruz?",
                 fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.4, hspace=0.5)

    # BatchNorm: dağılım karşılaştırması
    ax1 = fig.add_subplot(gs[0, 0])
    np.random.seed(0)
    ham_aktivasyon = np.random.normal(loc=5.0, scale=3.0, size=1000)
    ax1.hist(ham_aktivasyon, bins=40, color='#E74C3C', alpha=0.7)
    ax1.set_title("BatchNorm Öncesi\nAktivasyonlar (μ≈5, σ≈3)", fontsize=9)
    ax1.set_xlabel("Aktivasyon Değeri"); ax1.set_ylabel("Frekans")
    ax1.axvline(ham_aktivasyon.mean(), color='black', linestyle='--', lw=1.5, label=f'μ={ham_aktivasyon.mean():.1f}')
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    normalize = (ham_aktivasyon - ham_aktivasyon.mean()) / ham_aktivasyon.std()
    ax2.hist(normalize, bins=40, color='#2ECC71', alpha=0.7)
    ax2.set_title("BatchNorm Sonrası\nAktivasyonlar (μ≈0, σ≈1)", fontsize=9)
    ax2.set_xlabel("Aktivasyon Değeri")
    ax2.axvline(0, color='black', linestyle='--', lw=1.5, label='μ=0')
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    epoch_sayisi = 20
    without_bn = 0.001 * np.exp(-np.linspace(0, 2, epoch_sayisi)) + np.random.normal(0, 0.0001, epoch_sayisi)
    with_bn    = 0.001 * np.exp(-np.linspace(0, 4, epoch_sayisi)) + np.random.normal(0, 0.00005, epoch_sayisi)
    ax3.plot(without_bn, 'r-o', markersize=4, label='BatchNorm Yok')
    ax3.plot(with_bn,    'g-o', markersize=4, label='BatchNorm Var')
    ax3.set_title("BatchNorm → Daha Hızlı Yakınsama", fontsize=9)
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Learning Rate")
    ax3.legend(fontsize=8)

    # Dropout: nöron maskeleme
    ax4 = fig.add_subplot(gs[1, 0])
    n_noron = 10
    aktif   = np.ones((1, n_noron))
    im4 = ax4.imshow(aktif, cmap='Greens', aspect='auto', vmin=0, vmax=1)
    ax4.set_title("Dropout Öncesi — Tüm Nöronlar Aktif\n(10 nöron)", fontsize=9)
    ax4.set_yticks([]); ax4.set_xticks(range(n_noron))
    ax4.set_xticklabels([f'N{i+1}' for i in range(n_noron)], fontsize=7)
    for i in range(n_noron):
        ax4.text(i, 0, '✓', ha='center', va='center', color='darkgreen', fontsize=11, fontweight='bold')

    ax5 = fig.add_subplot(gs[1, 1])
    np.random.seed(42)
    mask = np.random.binomial(1, 0.5, (1, n_noron)).astype(float)
    im5 = ax5.imshow(mask, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax5.set_title("Dropout Sırasında (p=0.5)\nRastgele %50 Kapatıldı", fontsize=9)
    ax5.set_yticks([]); ax5.set_xticks(range(n_noron))
    ax5.set_xticklabels([f'N{i+1}' for i in range(n_noron)], fontsize=7)
    for i in range(n_noron):
        sembol = '✓' if mask[0, i] == 1 else '✗'
        renk   = 'darkgreen' if mask[0, i] == 1 else 'red'
        ax5.text(i, 0, sembol, ha='center', va='center', color=renk, fontsize=11, fontweight='bold')

    ax6 = fig.add_subplot(gs[1, 2])
    n_epoch = 30
    train_overfit = np.linspace(0.5, 0.97, n_epoch) + np.random.normal(0, 0.01, n_epoch)
    val_overfit   = np.linspace(0.5, 0.68, 15).tolist() + \
                    np.linspace(0.68, 0.62, n_epoch-15).tolist() + \
                    np.random.normal(0, 0.01, n_epoch)
    val_overfit   = np.array(val_overfit[:n_epoch])
    train_dropout = np.linspace(0.5, 0.88, n_epoch) + np.random.normal(0, 0.01, n_epoch)
    val_dropout   = np.linspace(0.5, 0.82, n_epoch) + np.random.normal(0, 0.01, n_epoch)
    ax6.plot(train_overfit, 'r--',  label='Train (Dropout Yok)', lw=1.5)
    ax6.plot(val_overfit,   'r-',   label='Val (Dropout Yok)',   lw=1.5)
    ax6.plot(train_dropout, 'g--',  label='Train (Dropout Var)', lw=1.5)
    ax6.plot(val_dropout,   'g-',   label='Val (Dropout Var)',   lw=1.5)
    ax6.set_title("Dropout → Overfitting Önleme Etkisi", fontsize=9)
    ax6.set_xlabel("Epoch"); ax6.set_ylabel("Accuracy")
    ax6.legend(fontsize=7); ax6.set_ylim(0.4, 1.05)

    plt.savefig("batchnorm_dropout_aciklamasi.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(" Kaydedildi: batchnorm_dropout_aciklamasi.png")

goster_batchnorm_dropout_aciklamasi()

#ODEL EGİTİM

model.compile(
    optimizer = Adam(learning_rate=0.001),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

# Callbacks 
# EarlyStopping : val_loss patience=5 epoch iyileşmezse dur,
#                 restore_best_weights ile en iyi ağırlıkları geri yükle.
# ReduceLROnPlateau: val_loss 3 epoch plateau → LR×0.2 küçül (min 1e-5).
early_stop = EarlyStopping(
    monitor             = 'val_loss',
    patience            = 5,
    restore_best_weights= True,
    verbose             = 1
)

reduce_lr = ReduceLROnPlateau(
    monitor  = 'val_loss',
    factor   = 0.2,
    patience = 3,
    min_lr   = 1e-5,
    verbose  = 1
)

history = model.fit(
    train_generator,
    validation_data = validation_generator,
    epochs          = EPOCHS,
    callbacks       = [early_stop, reduce_lr]
)

model.save("emotion_cnn_model.h5")
print(" Model kaydedildi: emotion_cnn_model.h5")

test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
print(f"\n Test Loss    : {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


#OVERFITTING ANALIZI
def goster_overfitting_analizi(history):
    """
    Train vs Validation accuracy ve loss eğrilerini çizer.

    Overfitting işaretleri:
      - Train accuracy sürekli artar, val accuracy durur/düşer
      - İki eğri arasındaki makas zamanla büyür
      - Loss için tersi: val_loss yükselirken train_loss düşer
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Overfitting Analizi — Eğitim Süreci İzleme",
                 fontsize=14, fontweight='bold')

    epochs_ran = range(1, len(history.history['accuracy']) + 1)

    #ACCURAY
    ax = axes[0]
    ax.plot(epochs_ran, history.history['accuracy'],     'b-o', markersize=4, label='Train Accuracy')
    ax.plot(epochs_ran, history.history['val_accuracy'], 'r-o', markersize=4, label='Val Accuracy')
    ax.fill_between(
        epochs_ran,
        history.history['accuracy'],
        history.history['val_accuracy'],
        alpha=0.15, color='orange',
        label='Overfitting Bölgesi'
    )
    ax.set_title("Model Accuracy\n(Aradaki fark büyürse overfitting!)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3)

    #LOSS
    ax = axes[1]
    ax.plot(epochs_ran, history.history['loss'],     'b-o', markersize=4, label='Train Loss')
    ax.plot(epochs_ran, history.history['val_loss'], 'r-o', markersize=4, label='Val Loss')
    ax.fill_between(
        epochs_ran,
        history.history['loss'],
        history.history['val_loss'],
        alpha=0.15, color='orange'
    )
    ax.set_title("Model Loss\n(Val loss yükselirse overfitting!)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # GAP ANALIZI
    ax = axes[2]
    acc_gap  = np.array(history.history['accuracy']) - np.array(history.history['val_accuracy'])
    loss_gap = np.array(history.history['val_loss']) - np.array(history.history['loss'])
    ax.plot(epochs_ran, acc_gap,  'g-o', markersize=4, label='Accuracy Gap (Train−Val)')
    ax.plot(epochs_ran, loss_gap, 'm-o', markersize=4, label='Loss Gap (Val−Train)')
    ax.axhline(y=0.10, color='red', linestyle='--', lw=1.5, label='Tehlike Eşiği (%10)')
    ax.fill_between(epochs_ran, 0, acc_gap, alpha=0.1, color='green')
    ax.set_title("Overfitting Gap Analizi\n(Kırmızı çizgiyi geçerse tehlike!)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Gap")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("overfitting_analizi.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(" Kaydedildi: overfitting_analizi.png")

    # Yorum
    son_acc_gap = history.history['accuracy'][-1] - history.history['val_accuracy'][-1]
    print(f"\n Son Epoch Accuracy Gap: {son_acc_gap:.4f}")
    if son_acc_gap > 0.10:
        print("  UYARI: Overfitting belirtisi var! (Gap > %10)")
    else:
        print(" İyi: Overfitting belirtisi yok. (Gap ≤ %10)")

goster_overfitting_analizi(history)


# FEATURE MAP GÖRSELLEŞTİRME 


def goster_feature_maps(model, test_generator, n_filtre=16):
    """
    Bir test görseli alır ve her Conv2D katmanının
    ilk n_filtre feature map'ini yan yana gösterir.

    Yorumlama kılavuzu:
      - Conv Blok 1: Yatay/dikey kenarlar, köşe noktaları
      - Conv Blok 2: Göz çevresi, burun, ağız bölgeleri
      - Conv Blok 3: Tam yüz ifadeleri (gülümseme, kaş çatma...)
    """
    # Test generatöründen bir görsel al
    batch_imgs, batch_labels = next(test_generator)
    ornek_gorsel = batch_imgs[0:1]          # (1, 48, 48, 1)
    gercek_sinif = class_labels[np.argmax(batch_labels[0])]

    # Conv katmanlarını bul
    conv_katmanlari = [l for l in model.layers if 'conv' in l.name]

    # Her conv katmanı için ayrı aktivasyon modeli
    fig_list = []
    for idx, katman in enumerate(conv_katmanlari):
        aktivasyon_modeli = Model(
            inputs  = model.input,
            outputs = katman.output
        )
        aktivasyonlar = aktivasyon_modeli.predict(ornek_gorsel, verbose=0)
        # aktivasyonlar.shape: (1, H, W, n_filters)

        n_goster = min(n_filtre, aktivasyonlar.shape[-1])
        n_sat = 2
        n_sut = n_goster // n_sat

        fig, axes = plt.subplots(n_sat, n_sut, figsize=(n_sut * 1.8, n_sat * 2))
        fig.suptitle(
            f"Feature Maps — {katman.name}  "
            f"(Çıktı: {aktivasyonlar.shape[1]}×{aktivasyonlar.shape[2]}×{aktivasyonlar.shape[3]})\n"
            f"Gerçek Sınıf: {gercek_sinif}",
            fontsize=11, fontweight='bold'
        )

        for i, ax in enumerate(axes.flat):
            if i < n_goster:
                harita = aktivasyonlar[0, :, :, i]
                ax.imshow(harita, cmap='viridis')
                ax.set_title(f'Filtre {i+1}', fontsize=7)
            ax.axis('off')

        plt.tight_layout()
        fname = f"feature_map_{katman.name}.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.show()
        print(f" Kaydedildi: {fname}")

    # Orijinal görsel + tüm katmanların karşılaştırması
    fig, axes = plt.subplots(1, len(conv_katmanlari) + 1,
                              figsize=(4 * (len(conv_katmanlari) + 1), 4))
    fig.suptitle("Feature Map Özeti — Soyutlama Hiyerarşisi",
                 fontsize=13, fontweight='bold')

    axes[0].imshow(ornek_gorsel[0].squeeze(), cmap='gray')
    axes[0].set_title(f"Girdi\n{gercek_sinif}\n48×48×1", fontsize=9)
    axes[0].axis('off')

    for idx, katman in enumerate(conv_katmanlari):
        aktivasyon_modeli = Model(inputs=model.input, outputs=katman.output)
        akt = aktivasyon_modeli.predict(ornek_gorsel, verbose=0)
        # Ortalama feature map göster
        ortalama_harita = akt[0].mean(axis=-1)
        axes[idx + 1].imshow(ortalama_harita, cmap='viridis')
        axes[idx + 1].set_title(
            f"{katman.name}\n"
            f"{akt.shape[1]}×{akt.shape[2]}×{akt.shape[3]}\n"
            f"({['Düşük','Orta','Yüksek'][idx]} Seviye)",
            fontsize=9
        )
        axes[idx + 1].axis('off')

    plt.tight_layout()
    plt.savefig("feature_map_ozet.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(" Kaydedildi: feature_map_ozet.png")

goster_feature_maps(model, test_generator, n_filtre=16)


# TAHMİN ve  DEĞERLENDİRME 

def goster_confusion_matrix(model, test_generator):
    predictions = model.predict(test_generator, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    cm = confusion_matrix(y_true, y_pred)

    # Normalize edilmiş versiyon
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Confusion Matrix", fontsize=14, fontweight='bold')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels, ax=ax1)
    ax1.set_title("Ham Sayılar")
    ax1.set_xlabel("Tahmin Edilen"); ax1.set_ylabel("Gerçek")

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=class_labels, yticklabels=class_labels, ax=ax2,
                vmin=0, vmax=1)
    ax2.set_title("Normalize (Satır Bazlı Recall)")
    ax2.set_xlabel("Tahmin Edilen"); ax2.set_ylabel("Gerçek")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(" Kaydedildi: confusion_matrix.png")

    print("\n Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_labels))

goster_confusion_matrix(model, test_generator)


# Tek Görüntü Tahmini
def gorsel_tahmin_et(model, img_path, class_labels):
    """
    Tek bir görüntüyü yükler, tahmin eder ve
    tüm sınıf olasılıklarını bar grafik olarak gösterir.
    """
    img = load_img(img_path, color_mode='grayscale',
                   target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction     = model.predict(img_array, verbose=0)[0]
    tahmin_idx     = np.argmax(prediction)
    tahmin_sinif   = class_labels[tahmin_idx]
    guven          = prediction[tahmin_idx] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Tahmin: {tahmin_sinif}  |  Güven: %{guven:.1f}",
                 fontsize=13, fontweight='bold', color='green')

    ax1.imshow(img_array[0].squeeze(), cmap='gray')
    ax1.set_title(f"{tahmin_sinif}\n%{guven:.1f}", fontsize=12)
    ax1.axis('off')

    renkler = ['#E74C3C' if i == tahmin_idx else '#3498DB'
               for i in range(len(class_labels))]
    bars = ax2.barh(class_labels, prediction * 100, color=renkler)
    ax2.set_xlabel("Olasılık (%)")
    ax2.set_title("Sınıf Olasılıkları (Softmax Çıkışı)")
    ax2.set_xlim(0, 110)
    for bar, val in zip(bars, prediction * 100):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f'%{val:.1f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig("tahmin_sonucu.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n Tahmin Edilen Duygu : {tahmin_sinif}")
    print(f"Güven Oranı         : %{guven:.2f}")

# test
img_path = "test_image.jpg"
if os.path.exists(img_path):
    gorsel_tahmin_et(model, img_path, class_labels)
else:
    print(f"\n  '{img_path}' bulunamadı.")
    print("   gorsel_tahmin_et(model, 'yuzun.jpg', class_labels) ile çağırabilirsin.")


def goster_ve_kaydet_final_results(history, model, test_dir, class_labels):
    """
     'result.png' dosyasını oluşturur ve 
    belirlenen klasördeki örnekleri tahmin et.
    """
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # Sol Üst: Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history.history['accuracy'], 'b-o', label='Eğitim Doğruluğu')
    ax1.plot(history.history['val_accuracy'], 'r-o', label='Test (Val) Doğruluğu')
    ax1.set_title('Eğitim Süreci: Doğruluk (Accuracy)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Sağ Üst: Loss
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history.history['loss'], 'b-o', label='Eğitim Kaybı')
    ax2.plot(history.history['val_loss'], 'r-o', label='Test (Val) Kaybı')
    ax2.set_title('Eğitim Süreci: Kayıp (Loss)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Alt Panel: Rastgele Test Tahminleri
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    # Test klasöründen rastgele 4 resim seç
    test_resimler = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                test_resimler.append(os.path.join(root, file))
    
    if len(test_resimler) >= 4:
        secilenler = np.random.choice(test_resimler, 4, replace=False)
        for i, img_path in enumerate(secilenler):
            img = load_img(img_path, color_mode='grayscale', target_size=(48, 48))
            img_arr = img_to_array(img) / 255.0
            pred = model.predict(np.expand_dims(img_arr, axis=0), verbose=0)
            tahmin_idx = np.argmax(pred)
            
            # Klasör isminden gerçek etiketi çek (dataset/test/angry/img.jpg -> angry)
            gercek_sinif = os.path.basename(os.path.dirname(img_path))
            tahmin_sinif = class_labels[tahmin_idx]
            
            sub_ax = fig.add_subplot(2, 4, 5 + i)
            sub_ax.imshow(img_arr.squeeze(), cmap='gray')
            renk = 'green' if tahmin_sinif == gercek_sinif else 'red'
            sub_ax.set_title(f"Gerçek: {gercek_sinif}\nTahmin: {tahmin_sinif}\nGüven: %{np.max(pred)*100:.1f}", 
                             color=renk, fontsize=10)
            sub_ax.axis('off')

    plt.tight_layout()
    plt.savefig("result.png", dpi=200)
    plt.show()

goster_ve_kaydet_final_results(history, model, test_dir, class_labels)