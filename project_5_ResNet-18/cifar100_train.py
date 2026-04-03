"""
CIFAR-100, her biri 32x32 piksel boyutunda 60.000 renkli görselden oluşan; 
20 ana kategori (superclasses) altında toplanmış 100 farklı alt sınıfa
 (elma, akvaryum balığı, kunduz vb.) sahip, modelin ince detayları ayırt etme yeteneğini ölçen akademik bir derin öğrenme veri setidir (Krizhevsky, 2009).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

batch_size = 128
learning_rate = 0.001
num_epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    progress_bar = tqdm(loader, desc="Eğitim")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader), 100 * correct / total

def test_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Test")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss / len(loader), 100 * correct / total

if __name__ == "__main__":
    print(f"Cihaz: {device} | Batch Size: {batch_size} | Epoch: {num_epochs}")

    #veri hazırlama
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    #model oluşturma(sıfırdan bir model yerine, önceden eğitilmiş bir ResNet18 kullanarak transfer öğrenme yaptık)
    model = resnet18(weights='IMAGENET1K_V1')

    # Tam bağlantılı (FC) katmanını Dropout ile yeniden tanımlandı  çünkü aşırı öğrenme ile karşılaşıldı
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
    nn.Dropout(0.5),          # %50 Dropout: Ezberlemeyi (Overfitting) engeller
    nn.Linear(num_ftrs, 100)  # CIFAR-100 için 100 çıkış sınıfı
)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    #egitim döngüsü
    train_losses, train_accs, test_losses, test_accs = [], [], [], []

    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = test_epoch(model, test_loader, criterion, device)
        
        scheduler.step()
        
        train_losses.append(tr_loss); train_accs.append(tr_acc)
        test_losses.append(te_loss); test_accs.append(te_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {tr_acc:.2f}% | Test Acc: {te_acc:.2f}%")

    torch.save(model.state_dict(), 'my_cifar100_model.pth')
    print("\nModel kaydedildi.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_accs, label='Eğitim'); ax1.plot(test_accs, label='Test')
    ax1.set_title('Accuracy (%)'); ax1.legend()
    
    ax2.plot(train_losses, label='Eğitim'); ax2.plot(test_losses, label='Test')
    ax2.set_title('Loss'); ax2.legend()
    
    plt.tight_layout()
    plt.savefig('my_cifar100_results.png')
    print("Grafikler kaydedildi.")
    plt.show()