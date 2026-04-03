import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchvision import datasets # Sınıf isimlerini otomatik çekmek için eklendi
from PIL import Image
import matplotlib.pyplot as plt
import os

model_path = 'my_cifar100_model.pth' 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sınıf isimlerini CIFAR-100 veri setinden otomatik çeker
try:
    temp_ds = datasets.CIFAR100(root='./data', train=False, download=True)
    class_names = temp_ds.classes
except:
    # İnternet yoksa veya hata oluşursa generic isimler kullan
    class_names = [f"Sınıf_{i}" for i in range(100)]

def load_trained_model(path, device):
    model = resnet18(weights=None)
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.5), 
        nn.Linear(num_ftrs, 100)
    )
    
    # weights_only=True kullanımı güvenli ve gereksiz parametreleri yüklemeyi engeller, bu da bellek kullanımını azaltır
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval() # Bu satır çok önemli, Dropout'u test modunda kapatır.
    return model

def predict_local_image(image_path, model, device, classes):
    try:
        img = Image.open(image_path).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, pred = torch.max(probs, 0)

        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        # Sınıf ismi İngilizce gelebilir (apple, mushroom vb.)
        plt.title(f"Tahmin: {classes[pred]} (%{conf.item()*100:.2f})")
        plt.axis('off')
        plt.show()
        
        print(f"Sonuç: {classes[pred]} (Güven: %{conf.item()*100:.2f})")

    except Exception as e:
        print(f"Hata: Resim işlenemedi. Detay: {e}")

if __name__ == "__main__":
    print(f"Model yükleniyor: {model_path}...")
    my_model = load_trained_model(model_path, device)
    
    test_image = "indir.jpg" 
    
    if os.path.exists(test_image):
        print(f"'{test_image}' üzerinde CIFAR-100 tahmini yapılıyor...")
        predict_local_image(test_image, my_model, device, class_names)
    else:
        print(f"Hata: '{test_image}' bulunamadı. Klasöre bir test resmi koy.")