# PROJE FİNAL RAPORU: AKILLI SES ANALİZ VE DOĞRULAMA SİSTEMİ

**Sürüm:** [v16.0](https://img.shields.io/badge/version-16.0-brightgreen)  
**Hazırlayan:** Mehmet Karataş  
**Öğrenci No:** 222523002  
**Kurum:** İskenderun Teknik Üniversitesi (İSTE)  

---

## 1. Proje Özeti
Bu proje, modern ses işleme teknolojileri ile yapay zekayı (OpenAI Whisper, GPT-4o ve Google Gemini) harmanlayan profesyonel bir masaüstü uygulamasıdır. Kullanıcıların sesli girdilerini %99 doğrulukla metne dönüştürür, bu verileri akademik ve kurumsal standartlarda analiz eder, duygu durumu tespiti yapar ve görsel grafiklerle desteklenmiş kapsamlı raporlar (PDF/Word) üretir.

## 2. Kullanılan Sistemler ve Teknolojiler

### 2.1. Ses İşleme ve Transkripsiyon
- **OpenAI Whisper (Hibrit Entegrasyon):** Ses kayıtlarını metne dönüştürür. Donanıma göre **NVIDIA CUDA (GPU)** veya CPU optimizasyonu yaparak 10 kata kadar hız artışı sağlar.
- **Auto-VAD (Voice Activity Detection):** Sessiz sahneleri otomatik algılayarak kaydı optimize eder.
- **FFmpeg:** Ses formatlarını (wav, mp3) dönüştürmek için kullanılan temel kütüphane.
- **SoundDevice & SoundFile:** Gerçek zamanlı yüksek kaliteli ses kaydı.

### 2.2. Yapay Zeka Analiz ve Etkileşim
- **GPT-4o & Gemini 1.5 Flash:** Dünyanın en güçlü iki dil modelini kullanarak derinlemesine içerik ve duygu analizi yapar.
- **Persona Sistemi:** Profesyonel Analist, Sert Mentor veya Teknoloji Gurusu gibi farklı AI kişilikleriyle etkileşim imkanı sunar.
- **AI Dil Koçu (Language Coach):** Yabancı dil öğrenenler için gramer düzeltme, kelime dağarcığı geliştirme ve mentorluk desteği sağlar.
- **OpenAI TTS (Text-to-Speech):** AI yanıtlarının doğal bir sesle (Onyx, Nova, vb.) kullanıcıya okunmasını sağlar.

### 2.3. Görselleştirme ve Raporlama
- **Dinamik Spektrum:** Ses dalgalarını gerçek zamanlı izleyen neon bar görselleştiricisi.
- **Sentiment Timeline:** Zaman bazlı duygu değişimini gösteren etkileşimli grafikler.
- **WordCloud & Matplotlib:** Kelime yoğunluğu ve normalize edilmiş duygu analizi grafiklerini üretir.
- **Çoklu Format Desteği:** Raporları hem **PDF** hem de düzenlenebilir **Word (.docx)** formatında dışa aktarma.

---

## 3. Uygulama Mimarisi ve Dosya Yapısı
Proje, yüksek modülarite ve "Cyberpunk Glassmorphism" tasarım felsefesiyle inşa edilmiştir:
- `gui.py`: CustomTkinter tabanlı modern kullanıcı arayüzü ve ana kontrol mantığı.
- `transcriber.py`: Whisper motoru ve asenkron transkripsiyon yönetimi.
- `analytics.py`: Duygu analizi ve grafik üretim motoru.
- `report_generator.py`: PDF ve AI tabanlı rapor üretim modülü.
- `gemini_client.py`: Google Gemini API entegrasyon katmanı.
- `audio_recorder.py`: VAD destekli akıllı ses kayıt yönetimi.
- `visualizer.py`: Gerçek zamanlı ses spektrumu görselleştirme.

---

## 4. Kurulum ve Çalıştırma

### 4.1. Donanım Gereksinimleri
- Python 3.10+
- NVIDIA GPU (Önerilen) veya modern bir CPU
- FFmpeg (Sistem yolunda ekli olmalıdır)

### 4.2. Başlatma
1. Bağımlılıkları yükleyin: `pip install -r requirements.txt`
2. Uygulamayı çalıştırın: `python main.py`
3. Ayarlar sekmesinden API anahtarlarınızı (.env olarak saklanır) yapılandırın.

---

## 5. Sonuç
Bu sistem, insan sesini yapay zeka ile anlamlandırarak verimliliği artıran, özellikle akademik raporlama ve dil eğitimi süreçlerinde fark yaratan bütünleşik bir çözüm sunmaktadır. Hibrit model yapısı sayesinde hem yerel cihaz gücünü hem de bulut tabanlı AI yeteneklerini en üst seviyede kullanır.
