"""
elektrooptik  sensör: kamera ile 360 derecelik tarama gerçekleştirliyor, trafik kurallarında trafik işaretlerinin tanınması
otonom araçların en temel gorevi cevreyi tanimak: isaretler( trafik levhaları)
 Yolo(You Only Look Once) algoritması: cnn tabanlı, tek bir sinir ağı kullanarak görüntüyü bölümlere ayırır ve her bölümdeki nesneleri tanır.
Yolo algoritması, nesne tanıma görevlerinde yüksek doğruluk ve hızlı işlem süresi sağlar. Bu algoritma, özellikle gerçek zamanlı uygulamalarda tercih edilir çünkü tek bir geçişte tüm nesneleri tanıyabilir.
Yolo algoritması, genellikle büyük veri setleri üzerinde eğitilir ve farklı nesne sınıflarını tanımak için kullanılır. Örneğin, trafik işaretlerini tanımak için Yolo algoritması, trafik işaretlerinin farklı türlerini içeren bir veri seti üzerinde eğitilebilir. Bu sayede, otonom araçlar trafikteki işaretleri hızlı ve doğru bir şekilde tanıyabilirler.
"""


from ultralytics import YOLO

#mmodeli seç yolov8n modeli
model= YOLO("yolov8n.pt")

model.train(
            data="traffic-sign-detection/data.yaml", #yaml dosyasında veri yollari ve sinif isimleri tanimli
            epochs=2, 
            batch=16, #mini batch boyutu donanima bagli ayarlanir
            imgsz=640,
            name="yolo_trafik_isaretleri",
            lr0=0.01, #baslangic ogrenme hizi, modelin ne kadar hizli ogrenmeye baslayacagini belirler
            optimizer="SGD",# ALTERNATİF OLARAK ADAM
            weight_decay=0.0005, #modelin genelleme yetenegini arttirmak icin kullanilir, modelin cok fazla ogrenmesini engeller
            momentum= 0.937, #SGD momentum 
            patience=50, #early stopping için sabır suresi,
            workers=2,#data loader worker sayisi,
            device="cpu",#cpu veya cuda
            save = True, #modelleri kaydet 
            save_period=1, #model kaydetme periyodu, her epoch sonunda modeli kaydeder
            val= True,#validation yap, her epoch sonunda validation yaparak modelin performansini degerlendirir
            verbose= True, #egitim surecinde detayli bilgi verir, her epoch sonunda kayit edilen modelin performansini ve diger metrikleri gosterir
            )
    
"""
terminalde egitim sırasında cıkan cıktıdaki metrikler:
box_loss: modelin tahmin ettiği sınırlayıcı kutuların gerçek kutulara ne kadar yakın olduğunu gösterir. Düşük bir box_loss değeri, modelin nesneleri doğru bir şekilde tanımladığını gösterir. (0.1- 0.3 )
cls_loss: modelin sınıflandırma hatalarını gösterir. Düşük bir cls_loss değeri, modelin nesneleri doğru sınıflara atadığını gösterir.( 1'in altına inmeli)
dfl_loss: modelin tahmin ettiği kutuların gerçek kutulara ne kadar benzer olduğunu gösterir. Düşük bir dfl_loss değeri, modelin nesneleri doğru bir şekilde tanımladığını gösterir.(0.5-1 civarında olsa yeterli)
ınstances:modelin kaç nesne tespit ettiğini gösterir. Bu, modelin ne kadar başarılı olduğunu değerlendirmek için önemli bir metriktir.

"""



