import torch
import numpy as np
from PIL import Image
import os
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from fastai.learner import load_learner
from safetensors.torch import load_file

# Gerekli sınıflar
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=None):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
    
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.view(x.size(0), -1)

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class EmotionResnet34(nn.Module):
    def __init__(self, num_classes=5):
        super(EmotionResnet34, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        self.head = nn.Sequential(
            AdaptiveConcatPool2d(),
            Flatten(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes, bias=False)
        )
        
    def _make_layer(self, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
            
        layers = []
        layers.append(BasicBlock(inplanes, planes, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        return x

print("Model Karşılaştırma Aracı")
print("-----------------------\n")

# Duygu sınıfları
emotions = []
if os.path.exists('model_classes.txt'):
    with open('model_classes.txt', 'r') as f:
        emotions = [line.strip() for line in f.readlines()]
else:
    emotions = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

print(f"Kullanılan duygu sınıfları: {emotions}")

# 1. FastAI modelini yükle
print("\n1. Orijinal PKL modelini yüklüyorum...")
try:
    pkl_path = 'optimized_emotion_classifier.pkl'
    learn = load_learner(pkl_path)
    fastai_model = learn.model
    fastai_model.eval()
    print("✅ FastAI model başarıyla yüklendi!")
except Exception as e:
    print(f"❌ FastAI model yüklenemedi: {e}")
    exit(1)

# 2. SafeTensor modelini yükle
print("\n2. SafeTensor modelini yüklüyorum...")
try:
    pytorch_model = EmotionResnet34(len(emotions))
    safetensor_path = 'emotion_resnet34.safetensors'
    state_dict = load_file(safetensor_path)
    pytorch_model.load_state_dict(state_dict, strict=False)
    pytorch_model.eval()
    print("✅ SafeTensors model başarıyla yüklendi!")
except Exception as e:
    print(f"❌ SafeTensors model yüklenemedi: {e}")
    exit(1)

# Görüntü işleme fonksiyonları
def preprocess_for_fastai(img):
    # FastAI modeli için ön işleme
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

def preprocess_for_pytorch(img):
    # PyTorch modeli için ön işleme
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# Test için rastgele görüntüler
def create_test_images():
    print("\n3. Test için görüntüler oluşturuluyor...")
    images = []
    
    # Basit test görüntüleri
    print("   - Basit test görüntüsü (beyaz kare)")
    img1 = np.zeros((48, 48, 3), dtype=np.uint8)
    img1[10:30, 10:30] = 255  # Beyaz kare
    images.append(("Basit test görüntüsü", Image.fromarray(img1).convert('RGB')))
    
    # Farklı renklerde alanlar içeren test görüntüsü
    print("   - Renkli test görüntüsü")
    img2 = np.zeros((48, 48, 3), dtype=np.uint8)
    img2[5:25, 5:25, 0] = 255  # Kırmızı bölge
    img2[15:35, 15:35, 1] = 255  # Yeşil bölge
    img2[25:45, 25:45, 2] = 255  # Mavi bölge
    images.append(("Renkli test görüntüsü", Image.fromarray(img2).convert('RGB')))
    
    # Gerçek bir görüntü varsa onu da yükleyelim
    try:
        import glob
        print("   - Gerçek test görüntüleri aranıyor...")
        img_files = glob.glob("*.jpg") + glob.glob("*.png")
        if img_files:
            for img_file in img_files[:2]:  # En fazla ilk 2 görüntüyü al
                print(f"   - Gerçek görüntü: {img_file}")
                img = Image.open(img_file).convert('RGB')
                images.append((f"Gerçek görüntü: {img_file}", img))
    except Exception as e:
        print(f"   - Gerçek görüntü yüklenemedi: {e}")
    
    return images

# Tahminleri karşılaştır
def compare_predictions(images):
    print("\n4. Tahminler karşılaştırılıyor...")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"   Cihaz: {device}")
    
    fastai_model.to(device)
    pytorch_model.to(device)
    
    for name, img in images:
        print(f"\n❯ Test görüntüsü: {name}")
        
        # FastAI modeli ile tahmin
        try:
            # FastAI'ın kendi tahmin fonksiyonunu kullanma
            fastai_pred = learn.predict(img)
            fastai_label = fastai_pred[0]
            fastai_probs = fastai_pred[2].numpy() if hasattr(fastai_pred[2], 'numpy') else fastai_pred[2]
            
            print(f"   FastAI tahmin: {fastai_label}")
            for i, prob in enumerate(fastai_probs):
                print(f"   - {emotions[i]}: {prob:.4f}")
        except Exception as e:
            print(f"   ❌ FastAI tahmin hatası: {e}")
        
        # PyTorch modeli ile tahmin
        try:
            input_tensor = preprocess_for_pytorch(img).to(device)
            
            with torch.no_grad():
                output = pytorch_model(input_tensor)
                probs = F.softmax(output, dim=1)[0].cpu()
            
            # En yüksek olasılıklı duyguyu seç
            _, predicted = torch.max(output, 1)
            pytorch_label = emotions[predicted.item()]
            
            print(f"   SafeTensor tahmin: {pytorch_label}")
            for i, prob in enumerate(probs):
                print(f"   - {emotions[i]}: {prob:.4f}")
                
            # Sonuçların karşılaştırılması
            if fastai_label == pytorch_label:
                print(f"   ✅ Sonuçlar eşleşiyor: {fastai_label}")
            else:
                print(f"   ❌ Sonuçlar farklı! FastAI: {fastai_label}, SafeTensor: {pytorch_label}")
        except Exception as e:
            print(f"   ❌ SafeTensor tahmin hatası: {e}")

# Ana işlem
if __name__ == "__main__":
    # Görüntü oluştur ve modellerle karşılaştır
    images = create_test_images()
    compare_predictions(images)
    
    print("\n5. Karşılaştırma tamamlandı!") 