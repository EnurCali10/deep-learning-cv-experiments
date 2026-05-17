"""
insan saga mi sola mi gittigini sayma  yolo kullanarak 
data = https://www.kaggle.com/datasets/khitthanhnguynphan/crowduit

"""
import cv2 #opencv
import numpy as np
from ultralytics import YOLO

#MODEL YÜKLEME
model = YOLO("yolov8n.pt")

#VİDEO DOSYASİNİN ACİLMASİ
cap = cv2.VideoCapture("2.mp4")

#1 frame oku ve video calisiyor mu diye bak
success, frame = cap.read()
if not success:
    exit("video calismiyor")
#YENİDEN BOYUTLANDİRMA
frame = cv2.resize(frame, (0,0), fx=0.6, fy=0.6)
frame_height, frame_width = frame.shape[:2]

#ORTAYA DİKEY CİZGİ, BU CİZGİYİ GECELERİ COUNT EDELİM 
line_x = int(frame_width *0.5)
offset = 10

#SAYACLAR
giren= 0
cikan=0
counted_ids = set()
person_last_x ={}

#YOLO İLE İNSAN SAYİMİ 
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (0,0), fx = 0.6, fy= 0.6)#yeniden boyutlandırma


    #results içerisinde bir sürü obje olabilir yani birden fazla obje track edilebilir
    results= model.track(frame, persist=True, stream=False, conf=0.25, iou=0.3, tracker="bytetrack.yaml")

    if results [0].boxes.id is not None :
        ids = results[0].boxes.id.int().tolist()#id ler 
        classes = results[0].boxes.cls.int().tolist()#siniflar
        xyxy  =results [0].boxes.xyxy #koordinatlar

        for i, box in enumerate(xyxy):
            cls_id = classes[i]
            track_id = ids[i]
            class_name = model.names[cls_id]

            if class_name != "person":
                continue # sadece insanlari al

        #koordinatlar
        x1, y1, x2, y2 = map(int, box)
        cx = int((x1+x2)/2)
        cy= int((y1+y2)/2)



        previoux_x = person_last_x.get(track_id, None)
        person_last_x[track_id]= cx


        if previoux_x is not None:
            #sagdan sola gecenler
            if previoux_x > line_x >=cx:
                if track_id not in counted_ids:
                    cikan +=1
                    counted_ids.add(track_id)

            elif previoux_x < line_x <= cx:
                if track_id not in counted_ids:
                    giren += 1
                    counted_ids.add(track_id)

            #kutulari cizdirme
            cv2.rectangle(frame, (x1,y1), (x2,y2),(0,255,0,3))
            cv2.putText(frame, f"{class_name} ID: {track_id}", (x1,y1 - 8),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)
            cv2.circle(frame, (cx,cy),4,(255,0,0),-1)




    #dikey ayrim cizgisi
    cv2.line(frame, (line_x,0),(line_x, frame_height), (0,0,255),2)

    #sayaclari ayarla
    cv2.putText(frame, f"Giren (saga):{giren}", (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    cv2.putText(frame, f"Cikan (sola): {cikan}", (10,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    #GORUNTUYU GOSTER 
    cv2.imshow("avm yon takip", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destoryAllWindows()