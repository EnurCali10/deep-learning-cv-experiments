import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import matplotlib.pyplot as plt
import os

model_path = 'my_cifar10_model.pth'
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ['Uçak', 'Araba', 'Kuş', 'Kedi', 'Geyik', 
               'Köpek', 'Kurbağa', 'At', 'Gemi', 'Kamyon']

def load_trained_model(path, device):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Hata: {path} bulunamadı! Önce eğitim yapmalısın.")
        
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval() #  Çıkarım modu
    return model

def predict_local_image(image_path, model, device, classes):
    try:
    
        img = Image.open(image_path).convert('RGB')
        
        # Ön İşleme (32x32 Resize ve Normalizasyon)
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(device)

        #  Model Tahmini
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, pred = torch.max(probs, 0)

        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"Tahmin: {classes[pred]} (%{conf.item()*100:.2f})")
        plt.axis('off')
        plt.show()
        
        print(f"Sonuç: {classes[pred]} (Güven: %{conf.item()*100:.2f})")

    except Exception as e:
        print(f"Hata: Resim işlenemedi. Detay: {e}")

if __name__ == "__main__":
    my_model = load_trained_model(model_path, device)
    
    test_image = "images.jpg" 
    
    if os.path.exists(test_image):
        print(f"'{test_image}' üzerinde tahmin yapılıyor...")
        predict_local_image(test_image, my_model, device, class_names)
    else:
        print(f"Hata: '{test_image}' dosyası bulunamadı. Lütfen klasöre bir resim ekle.")