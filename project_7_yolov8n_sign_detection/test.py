from ultralytics import YOLO
import cv2

#modeli yükle
model= YOLO("runs/detect/traffic-sign-model/weights/best.pt")

#test edilecek görsellerin yüklenmesi
image_path = "test/images/test_image.jpg" #test edilecek görselin yolu
image = cv2.imread(image_path)

results= model(image_path)[0]
print(results)

#kutu cizimi
for box in results.boxes:
    #koordinatlar
    x1, y1, x2, y2 = map(int, box.xyxy[0]) #kose koordinatları
    confidence = float(box.conf[0])#guven seviyesi
    class_id = int(box.cls[0])
    #kutu cizimi
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{model.names[class_id]}: {confidence:.2f}"
    #etiketi image üzerine yazdir
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
cv2.imshow("Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#kaydet
cv2.imwrite("prediction_result.jpg", image)
