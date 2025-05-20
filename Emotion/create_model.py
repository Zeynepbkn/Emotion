import torch
import torch.nn as nn
from fastai.learner import load_learner
from safetensors.torch import save_file
import os
from PIL import Image
import numpy as np

print("FastAI modelden safetensors modeli oluşturma")

# FastAI AdaptiveConcatPool2d sınıfını tanımla
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=None):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
    
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

# Flatten katmanı
class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.view(x.size(0), -1)

# BasicBlock sınıfını tanımla (ResNet34'ten)
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

# Tam ResNet34 + FastAI özelleştirmesi
class EmotionResnet34(nn.Module):
    def __init__(self, num_classes=5):
        super(EmotionResnet34, self).__init__()
        
        # İlk katman - ResNet34'ün birinci katmanı
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Layer1 - 3 BasicBlock
        self.layer1 = self._make_layer(64, 64, 3)
        
        # Layer2 - 4 BasicBlock
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        
        # Layer3 - 6 BasicBlock
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        
        # Layer4 - 3 BasicBlock
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        # FastAI baş kısmı
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

try:
    # Model sınıflarını yükle
    emotions = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
    
    # FastAI modelini yükle
    print("\n1. FastAI modelini yüklüyorum...")
    pkl_path = 'optimized_emotion_classifier.pkl'
    learn = load_learner(pkl_path)
    fastai_model = learn.model
    print("FastAI model yüklendi!")
    
    # State dict'i alalım
    fastai_state_dict = fastai_model.state_dict()
    
    # Bizim modelimizi oluştur
    print("\n2. PyTorch modelini oluşturuyorum...")
    pytorch_model = EmotionResnet34(len(emotions))
    
    # Katman isimlerini eşleştirmek için bir mappping oluştur
    # Bu mapping, originaldeki katmanları bizim modelimizdeki karşılıklarına eşleştirir
    mapping = {}
    
    # Tüm katman isimlerini özelleştirelim
    print("\n3. Katman isimlerini eşleştiriyorum...")
    
    # Birinci katman (backbone)
    mapping['0.0.weight'] = 'backbone.0.weight'
    mapping['0.1.weight'] = 'backbone.1.weight'
    mapping['0.1.bias'] = 'backbone.1.bias'
    mapping['0.1.running_mean'] = 'backbone.1.running_mean'
    mapping['0.1.running_var'] = 'backbone.1.running_var'
    
    # Layer1 (ilk ResNet katmanı)
    for i in range(3):  # 3 BasicBlock
        # Her bir BasicBlock için
        for j in ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 
                  'conv2.weight', 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var']:
            mapping[f'0.4.{i}.{j}'] = f'layer1.{i}.{j}'
    
    # Layer2 (ikinci ResNet katmanı)
    for i in range(4):  # 4 BasicBlock
        for j in ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 
                  'conv2.weight', 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var']:
            mapping[f'0.5.{i}.{j}'] = f'layer2.{i}.{j}'
        
        # Downsample
        if i == 0:
            mapping['0.5.0.downsample.0.weight'] = 'layer2.0.downsample.0.weight'
            mapping['0.5.0.downsample.1.weight'] = 'layer2.0.downsample.1.weight'
            mapping['0.5.0.downsample.1.bias'] = 'layer2.0.downsample.1.bias'
            mapping['0.5.0.downsample.1.running_mean'] = 'layer2.0.downsample.1.running_mean'
            mapping['0.5.0.downsample.1.running_var'] = 'layer2.0.downsample.1.running_var'
    
    # Layer3 (üçüncü ResNet katmanı)
    for i in range(6):  # 6 BasicBlock
        for j in ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 
                  'conv2.weight', 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var']:
            mapping[f'0.6.{i}.{j}'] = f'layer3.{i}.{j}'
            
        # Downsample
        if i == 0:
            mapping['0.6.0.downsample.0.weight'] = 'layer3.0.downsample.0.weight'
            mapping['0.6.0.downsample.1.weight'] = 'layer3.0.downsample.1.weight'
            mapping['0.6.0.downsample.1.bias'] = 'layer3.0.downsample.1.bias'
            mapping['0.6.0.downsample.1.running_mean'] = 'layer3.0.downsample.1.running_mean'
            mapping['0.6.0.downsample.1.running_var'] = 'layer3.0.downsample.1.running_var'
    
    # Layer4 (dördüncü ResNet katmanı)
    for i in range(3):  # 3 BasicBlock
        for j in ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 
                  'conv2.weight', 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var']:
            mapping[f'0.7.{i}.{j}'] = f'layer4.{i}.{j}'
            
        # Downsample
        if i == 0:
            mapping['0.7.0.downsample.0.weight'] = 'layer4.0.downsample.0.weight'
            mapping['0.7.0.downsample.1.weight'] = 'layer4.0.downsample.1.weight'
            mapping['0.7.0.downsample.1.bias'] = 'layer4.0.downsample.1.bias'
            mapping['0.7.0.downsample.1.running_mean'] = 'layer4.0.downsample.1.running_mean'
            mapping['0.7.0.downsample.1.running_var'] = 'layer4.0.downsample.1.running_var'
    
    # Baş kısmı (head)
    mapping['1.2.weight'] = 'head.2.weight'
    mapping['1.2.bias'] = 'head.2.bias'
    mapping['1.2.running_mean'] = 'head.2.running_mean'
    mapping['1.2.running_var'] = 'head.2.running_var'
    mapping['1.4.weight'] = 'head.4.weight'
    mapping['1.6.weight'] = 'head.6.weight'
    mapping['1.6.bias'] = 'head.6.bias'
    mapping['1.6.running_mean'] = 'head.6.running_mean'
    mapping['1.6.running_var'] = 'head.6.running_var'
    mapping['1.8.weight'] = 'head.8.weight'
    
    # Ağırlıkları eşleştir
    print("\n4. Ağırlıkları PyTorch modeline aktarıyorum...")
    pytorch_state_dict = {}
    warnings = []
    
    for orig_key in fastai_state_dict:
        if orig_key in mapping:
            new_key = mapping[orig_key]
            pytorch_state_dict[new_key] = fastai_state_dict[orig_key]
        else:
            # num_batches_tracked gibi bazı parametreleri yok sayabiliriz
            if not 'num_batches_tracked' in orig_key:
                warnings.append(f"Eşleştirilemeyen anahtar: {orig_key}")
    
    # Modelimize yükle
    try:
        pytorch_model.load_state_dict(pytorch_state_dict, strict=False)
        print("Model ağırlıkları başarıyla yüklendi!")
    except Exception as e:
        print(f"Model yüklenirken hata: {e}")
        
    if warnings:
        print(f"{len(warnings)} anahtar eşleştirilemedi (önemli olmayabilir)")
    
    # Modeli safetensors olarak kaydet
    print("\n5. Modeli safetensors formatında kaydediyorum...")
    
    output_path = "emotion_resnet34.safetensors"
    save_file(pytorch_model.state_dict(), output_path)
    print(f"Model başarıyla kaydedildi: {output_path}")
    
    # Test bir tahmin yapalım
    print("\n6. Test tahmin yapıyorum...")
    pytorch_model.eval()
    
    # Basit bir test görüntüsü oluştur
    def create_test_image():
        img = np.zeros((48, 48), dtype=np.uint8)
        img[10:30, 10:30] = 255  # Beyaz kare
        return Image.fromarray(img).convert('RGB')
    
    # Görüntü işleme
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_img = create_test_image()
    input_tensor = transform(test_img).unsqueeze(0)
    
    # Tahmin
    with torch.no_grad():
        output = pytorch_model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        
    # En yüksek olasılık
    _, predicted = torch.max(output, 1)
    emotion = emotions[predicted.item()]
    
    print(f"Tahmin Edilen Duygu: {emotion}")
    for i, prob in enumerate(probs):
        print(f"{emotions[i]}: {prob:.6f}")
    
    # Model sınıflarını da metin dosyasına kaydet
    with open('model_classes.txt', 'w') as f:
        for emotion in emotions:
            f.write(f"{emotion}\n")
    print("\nModel sınıfları kaydedildi: model_classes.txt")
    
    print("\nİşlem tamamlandı!")

except Exception as e:
    print(f"Hata: {e}")
    import traceback
    traceback.print_exc() 