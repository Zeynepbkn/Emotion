import torch
import os
import sys
from fastai.learner import load_learner
from PIL import Image
import numpy as np
import torch.nn as nn

# Model dosya yolu
PKL_MODEL_PATH = 'optimized_emotion_classifier.pkl'

print("FastAI model mimarisi analiz ediliyor...")
print(f"Yüklenen dosya: {PKL_MODEL_PATH}")

try:
    # Modeli yükle
    learn = load_learner(PKL_MODEL_PATH)
    print("\n=== Model başarıyla yüklendi! ===")
    
    # Model sınıflarını analiz et
    print("\n=== Model Sınıfları ===")
    if hasattr(learn, 'dls') and hasattr(learn.dls, 'vocab'):
        classes = learn.dls.vocab
        print(f"Sınıflar: {classes}")
    else:
        print("Sınıf bilgisi bulunamadı")
    
    # Modelin kendisini analiz et
    model = learn.model
    print("\n=== Model Tipi ===")
    print(f"Model tipi: {type(model)}")
    
    # Modelin yapısını analiz et
    print("\n=== Model Yapısı ===")
    print(model)
    
    # Katmanları detaylıca analiz et
    print("\n=== Katman Detayları ===")
    for name, layer in model.named_children():
        print(f"\nKatman Adı: {name}")
        print(f"Katman Tipi: {type(layer)}")
        print(f"Katman Yapısı: {layer}")
        
        # Alt katmanları da analiz et
        if hasattr(layer, 'named_children'):
            for sub_name, sub_layer in layer.named_children():
                print(f"  Alt Katman Adı: {sub_name}")
                print(f"  Alt Katman Tipi: {type(sub_layer)}")
                print(f"  Alt Katman Yapısı: {sub_layer}")
    
    # State dictionary'yi analiz et
    print("\n=== State Dictionary Anahtarları ===")
    state_dict = model.state_dict()
    
    print(f"Toplam parametre sayısı: {len(state_dict)}")
    
    # Anahtarları ve şekillerini yazdır
    print("\nKatman\t\t\t\tŞekil")
    print("-" * 60)
    for key, param in state_dict.items():
        print(f"{key}\t\t{param.shape}")
    
    # Giriş ve çıkış boyutlarını belirlemeye çalış
    print("\n=== Giriş ve Çıkış Özellikleri ===")
    
    # Giriş boyutunu tahmin et
    first_layer = None
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            first_layer = layer
            break
    
    if first_layer is not None:
        in_channels = first_layer.in_channels
        print(f"Giriş kanalları: {in_channels}")
    else:
        print("Giriş kanalları belirlenemedi")
    
    # Çıkış boyutunu tahmin et
    last_layer = None
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            last_layer = layer
    
    if last_layer is not None:
        out_features = last_layer.out_features
        print(f"Çıkış özellikleri: {out_features}")
    else:
        print("Çıkış özellikleri belirlenemedi")
    
    # Test tahmin için basit bir örnek oluştur
    print("\n=== Test Tahmin ===")
    
    # Basit test görüntüsü oluştur
    def create_test_image():
        # Basit bir test görüntüsü oluştur - siyah arka plan üzerine beyaz kare
        img = np.zeros((48, 48), dtype=np.uint8)
        img[10:30, 10:30] = 255  # Beyaz kare
        return Image.fromarray(img).convert('L')
    
    test_img = create_test_image()
    
    # Modelin prediction metodunu çağır
    pred = learn.predict(test_img)
    
    print(f"Tahmin (prediction[0]): {pred[0]}")
    print(f"Tahmin tipi: {type(pred[0])}")
    print(f"İkinci eleman (prediction[1]) tipi: {type(pred[1])}")
    print(f"Üçüncü eleman (prediction[2]) tipi: {type(pred[2])}")
    
    if hasattr(pred[2], 'numpy'):
        probs = pred[2].numpy()
        print(f"Olasılıklar (prediction[2]): {probs}")
    
    # Önişleme ve sonişleme metodlarını belirlemeye çalış
    print("\n=== Önişleme/Sonişleme Fonksiyonları ===")
    if hasattr(learn, 'dls') and hasattr(learn.dls, 'after_item'):
        print(f"Item dönüşümleri: {learn.dls.after_item}")
    if hasattr(learn, 'dls') and hasattr(learn.dls, 'before_batch'):
        print(f"Batch öncesi dönüşümler: {learn.dls.before_batch}")
    if hasattr(learn, 'dls') and hasattr(learn.dls, 'after_batch'):
        print(f"Batch sonrası dönüşümler: {learn.dls.after_batch}")
    
    # Model özeti için fastai learn.summary() metodunu çağır
    try:
        print("\n=== Model Özeti (FastAI) ===")
        
        # Inner forward metodu ekleyelim tüm işlemleri görmek için
        def hook_inner_forward(mod, inp, out):
            print(f"Katman: {mod.__class__.__name__}")
            print(f"Giriş şekli: {inp[0].shape if isinstance(inp, tuple) and len(inp) > 0 else 'Bilinmiyor'}")
            print(f"Çıkış şekli: {out.shape if hasattr(out, 'shape') else 'Bilinmiyor'}")
            print("-" * 30)
        
        # Model özeti çıkar
        summary_hooks = []
        for layer in model.children():
            hook = layer.register_forward_hook(hook_inner_forward)
            summary_hooks.append(hook)
        
        # Dummy girdi oluştur
        dummy_input = torch.randn(1, 3, 48, 48)  # Varsayılan olarak 3 kanallı
        
        # Forward geçişi
        print("Dummy input ile forward geçişi:")
        try:
            with torch.no_grad():
                _ = model(dummy_input)
        except Exception as e:
            print(f"Forward geçişi sırasında hata: {e}")
            # 1 kanal için dene
            print("1 kanal ile deneniyor...")
            try:
                dummy_input = torch.randn(1, 1, 48, 48)
                with torch.no_grad():
                    _ = model(dummy_input)
            except Exception as e2:
                print(f"1 kanal ile de hata: {e2}")
        
        # Hook'ları kaldır
        for hook in summary_hooks:
            hook.remove()
            
    except Exception as e:
        print(f"Özet çıkarılırken hata: {e}")
    
    print("\n=== Analiz tamamlandı! ===")
    
except Exception as e:
    print(f"Hata: {e}")
    import traceback
    traceback.print_exc() 