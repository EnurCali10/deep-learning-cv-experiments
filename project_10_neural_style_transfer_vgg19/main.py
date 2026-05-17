"""
stil transferi: görselin içeriğini alıp başka bir görsel üzerine boyamak olarak tanımlanabilir. Bu tekniği kullanarak, bir görselin içeriğini koruyarak başka bir görselin stilini uygulayabilirsiniz. Örneğin, bir fotoğrafın içeriğini alıp, 
Van Gogh'un "Yıldızlı Gece" tablosunun stilini uygulayarak yeni bir görsel oluşturabilirsiniz.
sanatsal gorselleştirme,moda tasarımı , fotografcilik gibi alanlaarda kullanılabilir.

veri seti :content ve style gorselleri

"""
from matplotlib.pylab import shape, std
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
image_path: yuklenecek gorselin dosya yolu
max_size:gorselin maksimum pikseli uzun goruntuleri hiz kazandirmak ici px e kadar kucultuyoruz
shape: stil resmiyle ayni boyuta esitlemek icin (H,W) tuple
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path, max_size=400, shape=None):
    image = Image.open(image_path).convert('RGB')
    
    if shape is not None:
        size = shape #stil ve icerik aynı H x W
    else:
        size= max(image.size)# uzun kenari al
        if size > max_size: # fazla uzun ise kirp
            size = max_size

    #donusumler
    in_transform = transforms.Compose([
        transforms.Resize((size, size) if isinstance(size, int) else size),#yeniden boyutlandırma
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485,0.456,0.406],#her kanal icin ortalama 
        std=[0.229,0.224,0.225]#her kanal icin standart sapma
        )
    ])
    
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image.to(device)

#gorseli ekranda duzgun gosterme
def im_convert(tensor):
    """
    tensoru tekaradan 0-1 aralıgına ve (h,w,3) formuna cevirir cunku matplotlib ile gorsellestirme yapacagız 
    """
    image = tensor.clone().detach().cpu().squeeze(0)
    #std ile carp ve ortalama ile topla yani ters normalization yap
    image = image * torch.tensor([0.229,0.224,0.225]).view(3,1,1) + torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    image = image.clamp(0,1)
    return image.permute(1,2,0).numpy()# (h,w,3) formuna cevirme
    
#gramm matrisi yani stil benzerligi olcutu 
def gram_matrix(tensor):
    """
    (C,H,W) (C,H*W) = A, A X A. T = gram matris
    """
    _, d, h, w = tensor.size()
    tensor = tensor.view(d,h*w)
    return torch.mm(tensor, tensor.t())

#oznitelik cikarici model(vgg19)
class VGG(nn.Module):
    """
    imagenet ile pre trained yapilan vgg19dan hem icerik hem stiil bilgisini cekelim
    """
    def __init__(self):
        super(VGG, self).__init__()
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:29].to(device).eval()

        for param in self.vgg.parameters():
            param.requires_grad = False#agirlikları sabit tut 

        #katman isim eslestirmesi
        self.layers={
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "21": "conv4_2",#icerik bilgisi
            "28": "conv5_1"
        }

    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features

#stil transferi dongusunu tamamla
def run_style_transfer(content_img,
                       style_img,
                       steps=2000,
                       style_weight=1e6,
                       content_weight=1):  
    #hedef tensoru icerik gorselinden kopyala ve optimize edilecek hale getir
    target = content_img.clone().requires_grad_(True).to(device)                          
    optimizer = optim.Adam([target], lr=0.003)
    model = VGG()  

    # sabit feature'ları bir kere hesapla (optimizasyon)
    content_features = model(content_img)
    style_features = model(style_img)

    for step in tqdm(range(steps)):
        target_features = model(target)

        #icerik kaybi
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

        #secili her katman icin gram matrrisi uzakligi hesapla 
        style_loss = 0
        for layer in ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]:
            target_feature= target_features[layer]
            style_feature= style_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = gram_matrix(style_feature)
            layer_loss = torch.mean((target_gram - style_gram)**2)
            style_loss += layer_loss 

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward() #geri yayılım taget sensoru parametreleri guncelle 
        optimizer.step()

        if step % 500 == 0:
            print(f"Adım {step}, Toplam Kayıp: {total_loss.item():.4f}")

    return target

#uygulama 
content = load_image("content.jpg")
style = load_image("style.png",shape=tuple(content.shape[-2:]))

output = run_style_transfer(content, style, steps=2000)

plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.title("Icerik Gorseli")
plt.imshow(im_convert(content))
plt.subplot(1,3,2)
plt.title("Stil Gorseli")
plt.imshow(im_convert(style))
plt.subplot(1,3,3)
plt.title("Stil Transferi Sonucu")
plt.imshow(im_convert(output))
plt.show()