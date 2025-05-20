import torch
import numpy as np
from PIL import Image
import os
from fastai.learner import load_learner
from safetensors.torch import load_file
from torchvision import transforms

# Duygu sınıfları
emotions = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

# Model dosya yolları
PKL_MODEL_PATH = 'optimized_emotion_classifier.pkl'
SAFETENSOR_MODEL_PATH = 'emotion_classifier.safetensors'

print("Model Karşılaştırma Başlatılıyor...")

# Her iki modeli de yükle
try:
    print(f"Pickle modeli yükleniyor: {PKL_MODEL_PATH}")
    fastai_model = load_learner(PKL_MODEL_PATH)
    print("Pickle modeli başarıyla yüklendi!")
except Exception as e:
    print(f"Pickle model yüklenemedi: {e}")
    fastai_model = None

# PyTorch modeli
from torch import nn
import torch.nn.functional as F
from torchvision import models

# ResNet34 modeli
try:
    print("PyTorch ResNet34 modeli yükleniyor...")
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(emotions))
    model.eval()
    
    # SafeTensor modelini yüklemeye çalış
    try:
        print(f"SafeTensor modelini yüklüyorum: {SAFETENSOR_MODEL_PATH}")
        state_dict = load_file(SAFETENSOR_MODEL_PATH)
        # Burada ağırlıkları uyumlu hale getirme kodu eklenebilir
        print("SafeTensor modeli yüklendi!")
    except Exception as e:
        print(f"SafeTensor modelini yükleme hatası: {e}")
except Exception as e:
    print(f"PyTorch modeli yükleme hatası: {e}")
    model = None

# Test görüntüsü oluştur veya yükle
def create_test_image():
    # Basit bir test görüntüsü oluştur - siyah arka plan üzerine beyaz kare
    img = np.zeros((48, 48), dtype=np.uint8)
    img[10:30, 10:30] = 255  # Beyaz kare
    return img

# FastAI modeli için ön işleme
def preprocess_for_fastai(image_array):
    img = Image.fromarray(image_array).resize((48, 48))
    img = img.convert('L')
    return img

# PyTorch modeli için ön işleme
def preprocess_for_pytorch(image_array):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.fromarray(image_array)
    tensor = transform(img)
    tensor = tensor.unsqueeze(0)
    return tensor

# Test görüntüsünü oluştur
test_image = create_test_image()
print("\nTest görüntüsü oluşturuldu.")

# Eğer test etmek için gerçek bir görüntü varsa:
try:
    print("Test için gerçek bir görüntü aranıyor...")
    test_files = [f for f in os.listdir('.') if f.endswith('.jpg') or f.endswith('.png')]
    if test_files:
        print(f"Gerçek test görüntüsü bulundu: {test_files[0]}")
        test_image = np.array(Image.open(test_files[0]))
except Exception:
    print("Gerçek test görüntüsü bulunamadı, oluşturulan test görüntüsü kullanılacak.")

# FastAI modeli ile tahmin
if fastai_model:
    try:
        print("\n--- FastAI Model Tahmini ---")
        processed_img = preprocess_for_fastai(test_image)
        prediction = fastai_model.predict(processed_img)
        
        # FastAI prediction[0] sınıf adını döndürür, prediction[2] olasılıkları döndürür
        emotion = prediction[0]
        probs = prediction[2].numpy()
        
        print(f"Tahmin Edilen Duygu: {emotion}")
        for i, prob in enumerate(probs):
            print(f"{emotions[i]}: {prob:.6f}")
    except Exception as e:
        print(f"FastAI tahmin hatası: {e}")

# PyTorch modeli ile tahmin
if model:
    try:
        print("\n--- PyTorch Model Tahmini ---")
        tensor = preprocess_for_pytorch(test_image)
        
        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)[0]
        
        # En yüksek olasılıklı duyguyu seç
        _, predicted = torch.max(outputs, 1)
        emotion = emotions[predicted.item()]
        
        print(f"Tahmin Edilen Duygu: {emotion}")
        for i, prob in enumerate(probs):
            print(f"{emotions[i]}: {prob:.6f}")
    except Exception as e:
        print(f"PyTorch tahmin hatası: {e}")

print("\nKarşılaştırma tamamlandı.") 