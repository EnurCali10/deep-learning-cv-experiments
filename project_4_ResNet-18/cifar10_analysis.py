import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'my_cifar10_model.pth'
CLASS_NAMES = ['Uçak', 'Araba', 'Kuş', 'Kedi', 'Geyik', 
               'Köpek', 'Kurbağa', 'At', 'Gemi', 'Kamyon']

def plot_confusion_matrix(model, loader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Analiz"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Tahmin Edilen (Predicted)')
    plt.ylabel('Gerçek (Actual)')
    plt.title('CIFAR-10 Karışıklık Matrisi (Confusion Matrix)')
    
    output_name = 'confusion_matrix.png'
    plt.savefig(output_name, dpi=150)
    print(f"\nBaşarılı! Karışıklık matrisi kaydedildi: {output_name}")
    plt.show()

if __name__ == "__main__":
    print(f"Cihaz: {DEVICE}")

    #test seti
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    if not os.path.exists(MODEL_PATH):
        print(f"Hata: {MODEL_PATH} dosyası bulunamadı! Önce eğitim yapmalısın.")
    else:
        print(f"Model yükleniyor: {MODEL_PATH}...")
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        
        # Ağırlıkları güvenli modda yükle
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model = model.to(DEVICE)

        plot_confusion_matrix(model, test_loader, DEVICE, CLASS_NAMES)