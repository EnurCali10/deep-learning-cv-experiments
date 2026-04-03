# Project 9: Bidirectional People Counting with YOLOv8

Bu çalışma, kalabalık alanlarda insan trafiğini analiz etmek amacıyla geliştirilmiştir. Görüntü merkezine çizilen dikey bir hattı geçen bireylerin yönü (sağdan sola veya soldan sağa) tespit edilerek sayım yapılır.

##  Teknik Özellikler
- **Model:** YOLOv8n (Nano) - Yüksek hız ve düşük gecikme için.
- **Veri Seti:** Kaggle CrowdUIT.
- **Algoritma:** Her bir `track_id` için son X koordinatı saklanır ve çizgi (line_x) ile kıyaslanarak geçiş yönü belirlenir.



##  Green AI 
- **Hafif Model:** En küçük YOLO varyantı (Nano) kullanılarak **RTX 3050** üzerinde minimum güç tüketimi sağlanmıştır.
- **CPU Verimliliği:** `cv2.resize` (fx=0.6) kullanılarak işlem yükü azaltılmıştır.