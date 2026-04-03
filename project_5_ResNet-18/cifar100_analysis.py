import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'my_cifar100_model.pth' 

def plot_confusion_matrix(model, loader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Tahmin Toplanıyor"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    #  En Zorlanılan Sınıflar 
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    # F1-Skoruna göre sınıfları sırala
    sorted_classes = sorted(report.items(), key=lambda x: x[1]['f1-score'] if isinstance(x[1], dict) else 1)
    
    print("\n--- Modelin En Çok Zorlandığı 5 Sınıf ---")
    for cls_name, metrics in sorted_classes[:5]:
        print(f"{cls_name}: F1-Score: {metrics['f1-score']:.2f}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(24, 20)) # Daha da büyük yaparak okunabilirliği artır
    
    sns.heatmap(cm, annot=False, cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    plt.xlabel('Tahmin Edilen (Predicted)')
    plt.ylabel('Gerçek (Actual)')
    plt.title('CIFAR-100 Karışıklık Matrisi (Heatmap Analysis)')
    
    output_name = 'confusion_matrix_cifar100_final.png'
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"\nAnaliz tamamlandı ve kaydedildi: {output_name}")
    plt.show()

if __name__ == "__main__":
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    class_names = test_dataset.classes 

    if not os.path.exists(MODEL_PATH):
        print(f"Hata: {MODEL_PATH} bulunamadı!")
    else:
        model = resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(num_ftrs, 100)
        )
        
        print(f"Model yükleniyor: {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model = model.to(DEVICE)

        plot_confusion_matrix(model, test_loader, DEVICE, class_names)