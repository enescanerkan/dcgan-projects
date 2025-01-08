# DCGAN Implementations for Different Domains (Farklı Alanlar İçin DCGAN Uygulamaları

## Datasets

- **Food Images**: RGB food images from the [Kaggle Food-101 Dataset](https://www.kaggle.com/datasets/kmader/food41)  
- **Fashion Items**: Grayscale clothing, shoes, and bags from the [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)  
- **Mammography Images**: Grayscale medical imaging from the [VinDr-Mammo Dataset](https://vindr.ai/datasets/mammo) 

  
This repository contains implementations of Deep Convolutional Generative Adversarial Networks (DCGANs) for generating synthetic images across three different domains (Bu depo, üç farklı alanda sentetik görüntüler oluşturmak için Derin Evrişimli Üretici Çekişmeli Ağların (DCGAN) uygulamalarını içerir):

- Grayscale Fashion Items - clothing, shoes, bags (Gri Tonlamalı Moda Öğeleri - kıyafet, ayakkabı, çanta)
- RGB Food Images (RGB Yemek Görüntüleri)
- Grayscale Mammography Images (Gri Tonlamalı Mamografi Görüntüleri)


## 🔍 Overview (Genel Bakış)

This project implements DCGANs to generate synthetic images in three distinct domains (Bu proje, üç farklı alanda sentetik görüntüler oluşturmak için DCGAN'ları uygular). Each implementation is specifically tailored to its domain's characteristics (Her uygulama, kendi alanının özelliklerine göre özel olarak uyarlanmıştır):

1. **Fashion DCGAN** (Moda DCGAN):
   - Input: Fashion-MNIST dataset - clothing, shoes, bags (Girdi: Fashion-MNIST veri seti - kıyafet, ayakkabı, çanta)
   - Output: 64x64 grayscale images (Çıktı: 64x64 gri tonlamalı görüntüler)
   - Single channel processing (Tek kanal işleme)

2. **Food DCGAN** (Yemek DCGAN):
   - Input: Food-41 dataset (Girdi: Food-41 veri seti)
   - Output: 128x128 RGB images (Çıktı: 128x128 RGB görüntüler)
   - Multi-channel color processing (Çok kanallı renk işleme)

3. **Mammography DCGAN** (Mamografi DCGAN):
   - Input: Mammography dataset (Girdi: Mamografi veri seti)
   - Output: 64x64 grayscale images (Çıktı: 64x64 gri tonlamalı görüntüler)
   - Specialized for medical image characteristics (Tıbbi görüntü özelliklerine özel)

## 🏗️ Model Architectures (Model Mimarileri)

![DCGAN-Architecture-used-in-this-study-Numbers-at-the-bottom-of-each-layer-indicate-the](https://github.com/user-attachments/assets/5859b3ce-e7c4-4017-b8bc-3cf1d070ce0d)


### Generator Architecture (Üretici Mimarisi)
```
Input: (latent_dim x 1 x 1) (Girdi: (gizli_boyut x 1 x 1))
└── ConvTranspose2d layers (Ters Evrişim katmanları):
    ├── 512 x 4 x 4
    ├── 256 x 8 x 8
    ├── 128 x 16 x 16
    ├── 64 x 32 x 32
    └── Output channels x 64 x 64 (or 128 x 128 for Food DCGAN)
        (Çıktı kanalları x 64 x 64 (veya Yemek DCGAN için 128 x 128))
```

### Discriminator Architecture (Ayırıcı Mimarisi)
```
Input: (channels x image_size x image_size)
(Girdi: (kanallar x görüntü_boyutu x görüntü_boyutu))
└── Conv2d layers (Evrişim katmanları):
    ├── 64 x 32 x 32
    ├── 128 x 16 x 16
    ├── 256 x 8 x 8
    ├── 512 x 4 x 4
    └── 1 (final output) (son çıktı)
```

## ⚙️ Training Parameters (Eğitim Parametreleri)

### Common Parameters (Ortak Parametreler)
```python
latent_dim = 100 (gizli_boyut = 100)
num_epochs = 200 (dönem_sayısı = 200)
batch_size = 32/64 (grup_boyutu = 32/64)
learning_rate_g = 0.0002 (öğrenme_oranı_g = 0.0002)
learning_rate_d = 0.0001 (öğrenme_oranı_d = 0.0001)
beta1 = 0.5 → 0.9
beta2 = 0.999
```

### Domain-Specific Parameters (Alana Özgü Parametreler)

1. **Fashion DCGAN** (Moda DCGAN)
```python
image_size = 64 (görüntü_boyutu = 64)
channels = 1 (kanallar = 1)
batch_size = 64 (grup_boyutu = 64)
```

2. **Food DCGAN** (Yemek DCGAN)
```python
image_size = 128 (görüntü_boyutu = 128)
channels = 3 (kanallar = 3)
batch_size = 32 (grup_boyutu = 32)
```

3. **Mammography DCGAN** (Mamografi DCGAN)
```python
image_size = 64 (görüntü_boyutu = 64)
channels = 1 (kanallar = 1)
batch_size = 32 (grup_boyutu = 32)
```

## 📝 Implementation Details (Uygulama Detayları)

### Key Features (Temel Özellikler)

1. **Advanced Training Techniques** (Gelişmiş Eğitim Teknikleri)
   - Label smoothing - 0.9 for real, 0.1 for fake (Etiket yumuşatma - gerçek için 0.9, sahte için 0.1)
   - Gradient clipping - max_norm=1.0 (Gradyan kırpma - maksimum_norm=1.0)
   - Gaussian noise injection (Gauss gürültüsü enjeksiyonu)
   - Learning rate scheduling (Öğrenme oranı planlaması)

2. **Model Improvements** (Model İyileştirmeleri)
   - Spectral normalization (Spektral normalizasyon)
   - Dropout layers for regularization (Düzenlileştirme için dropout katmanları)
   - Batch normalization in both networks (Her iki ağda da grup normalizasyonu)

3. **Training Monitoring** (Eğitim İzleme)
   - Loss tracking for both networks (Her iki ağ için kayıp takibi)
   - Regular sample generation (Düzenli örnek üretimi)
   - Grid visualization of generated images (Üretilen görüntülerin ızgara görselleştirmesi)

## 🔧 Requirements (Gereksinimler)
```
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.2
Pillow>=8.0.0
matplotlib>=3.3.0
```

## 🚀 Usage (Kullanım)

1. Clone the repository (Depoyu klonlayın):
```bash
git clone https://github.com/enescanerkan/dcgan-projects.git
cd dcgan-projects
```

2. Install dependencies (Bağımlılıkları yükleyin):
```bash
pip install -r requirements.txt
```

3. Run training for specific domain (Belirli alan için eğitimi çalıştırın):

* All the training code is contained within the dc-gan-training.ipynb file.(Tüm eğitim kodları dc-gan-training.ipynb içinde yer almaktadır.) 


## 📊 Results (Sonuçlar)

## RGB Food Images

https://github.com/user-attachments/assets/d7699c9b-fe8e-4119-9e90-5ec674a2213d


## Grayscale Fashion Items

https://github.com/user-attachments/assets/8d8561d5-fcc0-4ec0-97a8-93cf36e0de45


## Grayscale Mammography Images 

https://github.com/user-attachments/assets/0bc52143-6a34-470a-95f6-0b3476073482





