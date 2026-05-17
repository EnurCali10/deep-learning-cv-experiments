import cv2
import numpy as np
from tensorflow.keras.models import load_model

model= load_model("mnist_cnn_model.h5")

cap = cv2.VideoCapture(0)
print("bir kagida siyah kalemle rakam yaz ve kameraya goster, cikmak icin q tusuna bas")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    box_size = 200
    top_left = (w // 2 - box_size // 2, h // 2 - box_size // 2)
    bottom_right = (w // 2 + box_size // 2, h // 2 + box_size // 2)
    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

    roi= gray[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    roi= cv2.resize(roi, (28, 28))#yeniden boyutlandırma
    roi= roi.reshape(1, 28, 28, 1)

    #tahmin yapma
    pred = model.predict(roi, verbose=0)
    digit = np.argmax(pred)

    #tahmin sonucunu ekrana yazdirma
    cv2.putText(frame, f"Predicted: {digit}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

