import os
import torch
from fastai.learner import load_learner
from safetensors.torch import save_file
import numpy as np

print("Model dönüştürme işlemi başlatılıyor...")

# Model dosya yolu
pickle_model_path = 'optimized_emotion_classifier.pkl'
safetensor_model_path = 'emotion_classifier.safetensors'

# Modeli pickle formatından yükle
try:
    print(f"'{pickle_model_path}' dosyası yükleniyor...")
    learn = load_learner(pickle_model_path)
    print("Model başarıyla yüklendi!")
    
    # FastAI modelinden PyTorch modelini çıkar
    model = learn.model
    
    # Model state dict'ini al
    state_dict = model.state_dict()
    
    # NumPy array'lerini tensörlere çevir (eğer gerekirse)
    for key in state_dict:
        if isinstance(state_dict[key], np.ndarray):
            state_dict[key] = torch.from_numpy(state_dict[key])
    
    # Safetensors formatında kaydet
    print(f"Model '{safetensor_model_path}' olarak dönüştürülüyor...")
    save_file(state_dict, safetensor_model_path)
    print(f"Model başarıyla safetensors formatında kaydedildi: {safetensor_model_path}")
    
    # Eğer varsa model sınıflarını da kaydet
    if hasattr(learn, 'dls') and hasattr(learn.dls, 'vocab'):
        # Sınıf adlarını bir metin dosyasına yazdırma
        classes = learn.dls.vocab
        with open('model_classes.txt', 'w') as f:
            for cls in classes:
                f.write(f"{cls}\n")
        print("Model sınıfları 'model_classes.txt' dosyasına kaydedildi.")
    
    print("\nDönüştürme işlemi tamamlandı!")
    print("Şimdi app.py dosyanızı safetensors formatını kullanacak şekilde güncelleyin.")
    
except Exception as e:
    print(f"Hata oluştu: {e}") 