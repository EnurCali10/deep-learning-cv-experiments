# Project 10: Traffic Flow Analysis & Vehicle Counting

 Bu çalışma, otoyol videoları üzerinde araç türü bazlı sayım yapar. Standart dikey/yatay çizgilerin aksine, perspektife uygun **çapraz çizgi** algoritması kullanılmıştır.

##  Teknik Detaylar
- **Model:** YOLOv8n.
- **Veri Seti:** Kaggle Car Detection Videos.
- **Geometrik Analiz:** `get_line_side` fonksiyonu ile nesne merkezinin (cx, cy) çizginin hangi tarafında olduğu matematiksel olarak hesaplanır ve taraf değişikliği "geçiş" olarak kaydedilir.



## Green AI 
Bu proje, akıllı şehir sistemlerinde düşük enerji tüketen  (RTX 3050 vb.) verimli trafik izleme yapılabileceğini gösterir.