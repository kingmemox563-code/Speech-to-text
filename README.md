# Akıllı Ses Analiz ve Doğrulama Sistemi

Bu proje, yapay zeka destekli bir ses kayıt, transkripsiyon (metne dönüştürme) ve detaylı analiz sistemidir. OpenAI Whisper modelini yerel transkripsiyon için, GPT-4o modelini ise derinlemesine metin analizi için kullanır.

## 🚀 Öne Çıkan Özellikler

- **Gerçek Zamanlı Ses Kaydı**: Mikrofon üzerinden yüksek kaliteli ses kaydı.
- **Whisper Transkripsiyon**: Ses dosyalarını otomatik olarak metne dönüştürür (Türkçe dahil 7+ dil desteği).
- **GPT-4o Analizi**: Transkript edilen metni; özet, ana konular, duygu analizi ve eylem planı olarak analiz eder.
- **Görsel Analitik**: Kelime bulutu (WordCloud) ve duygu durum grafikleri (Sentiment Chart).
- **Profesyonel PDF Raporlama**: Tüm analiz sonuçlarını ve grafikleri içeren kurumsal yapıda bir rapor oluşturur.
- **Donanım Uyumluluğu**: NVIDIA GPU (CUDA) ve CPU üzerinde optimize çalışma.

## 🛠 Kurulum ve Sistem Gereksinimleri

### 1. FFmpeg Kurulumu (Kritik)
Ses işleme için sisteminizde FFmpeg yüklü olmalıdır.
1. [ffmpeg.org](https://ffmpeg.org/download.html) adresinden indirin.
2. `bin` klasörünü sistem PATH'inize ekleyin.

### 2. Donanıma Göre Kurulum (PyTorch)

#### A. NVIDIA Ekran Kartınız Varsa (Önerilen)
En iyi performans için CUDA desteğiyle kurun:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### B. Sadece CPU Kullanıyorsanız
```bash
pip install torch torchvision torchaudio
```

### 3. Kütüphanelerin Yüklenmesi
```bash
pip install -r requirements.txt
```

## ⚙️ Yapılandırma

1. Uygulamayı çalıştırdığınızda yan paneldeki **OpenAI Key** alanına API anahtarınızı girin.
2. **Save Key** butonuna basarak kaydedin (Bu anahtar yerel `config.json` dosyasında saklanır).

## 📖 Kullanım

1. **Model Seçimi**: Siteminize göre model seçin:
   - **NVIDIA GPU (8GB+ VRAM)**: `medium` veya `large`
   - **Giriş Seviye GPU / İyi CPU**: `small`
   - **Zayıf Sistemler**: `tiny` veya `base`
2. **Kayda Başla**: "START RECORDING" butonuna basın, konuşun ve "STOP RECORDING" ile bitirin.
3. **Analiz Et**: Transkript oluştuktan sonra "ANALYZE WITH GPT-4o" butonuna basarak yapay zeka analizini başlatın.
4. **PDF Kaydet**: Sonuçları "SAVE AS PDF" butonuyla kurumsal bir rapora dönüştürün.

## 🎓 Proje Hakkında
Bu proje, **İskenderun Teknik Üniversitesi (İSTE)** bünyesinde gerçekleştirilen bir ders projesi kapsamında geliştirilmiştir.
- **Geliştirici**: Mehmet Karataş
- **Ders**: Ders Projesi Teslimi

Mühendislikte Bilgisayar Uygulamaları I Dersi kapsamında geliştirilmiştir
