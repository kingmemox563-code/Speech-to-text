# PROJE FİNAL RAPORU: AKILLI SES ANALİZ VE DOĞRULAMA SİSTEMİ

**Hazırlayan:** Mehmet Karataş  
**Öğrenci No:** 222523002  
**Ders:** MÜHENDİSLİKTE BİLGİSAYAR UYGULAMALARI I  

---

## 1. Proje Özeti
Bu proje, kullanıcıların sesli girdilerini yapay zeka desteğiyle işleyen, analiz eden ve profesyonel raporlara dönüştüren bir masaüstü uygulamasıdır. Sistem, modern doğal dil işleme (NLP) ve ses işleme teknolojilerini bir araya getirerek hem bireysel hem de akademik kullanım için güçlü bir araç sunar.

## 2. Kullanılan Sistemler ve Teknolojiler

### 2.1. Ses İşleme ve Transkripsiyon
- **OpenAI Whisper:** Ses kayıtlarını yüksek doğrulukla metne dönüştürmek için kullanılan temel modeldir. Uygulama, kullanıcının donanımına göre (CPU veya NVIDIA GPU) en uygun Whisper modelini otomatik olarak seçer.
- **FFmpeg:** Ses dosyalarının format dönüştürme ve işleme süreçleri için arka planda çalışan kritik bir kütüphanedir.
- **SoundDevice & SoundFile:** Gerçek zamanlı ses kaydı ve dosya yönetimi için kullanılmıştır.

### 2.2. Yapay Zeka Analiz Motorları
- **OpenAI GPT-4o:** Metinlerin derinlemesine analizi, duygu durumu tespiti, özetleme ve anahtar nokta çıkarımı için kullanılır.
- **Google Gemini 1.5 Flash:** Alternatif bir analiz motoru olarak entegre edilmiştir. Daha hızlı ve etkili metin işleme yetenekleri sunar.
- **OpenAI TTS (Text-to-Speech):** AI yanıtlarının doğal bir sesle kullanıcıya okunmasını sağlar.

### 2.3. Görselleştirme ve Raporlama
- **Matplotlib:** Duygu analizi verilerini pasta grafikleriyle görselleştirir.
- **WordCloud:** Metindeki en sık kullanılan kelimeleri görsel bir bulut olarak sunar.
- **FPDF:** Tüm analiz sonuçlarını, transkriptleri ve grafikleri içeren profesyonel bir PDF raporu oluşturur.

---

## 3. Uygulama Mimarisi ve Dosya Yapısı
Proje modüler bir yapıda tasarlanmıştır:
- `gui.py`: Kullanıcı arayüzü ve ana kontrol mantığı.
- `transcriber.py`: Whisper entegrasyonu ve transkripsiyon.
- `analytics.py`: Grafiklerin ve görsel analizlerin oluşturulması.
- `report_generator.py`: PDF rapor üretim motoru.
- `gemini_client.py`: Google Gemini API entegrasyonu.
- `audio_recorder.py`: VAD destekli ses kayıt yönetimi.

## 4. Kurulum ve Çalıştırma

### 4.1. Gereksinimler
- Python 3.10+
- FFmpeg (Sistem yolunda ekli olmalıdır)
- Gerekli kütüphaneler: `pip install -r requirements.txt`

### 4.2. API Anahtarları
Uygulamanın tam fonksiyonel çalışması için `.env` dosyasında geçerli **OpenAI** ve **Gemini** API anahtarları bulunmalıdır.

### 4.3. Çalıştırma
`python main.py` komutuyla uygulama başlatılır.

---

## 5. Sonuç
Proje, ses teknolojileri ile yapay zekayı harmanlayarak kullanıcıya anlamlı veriler sunan, raporlanabilir ve kullanıcı dostu bir sistem olarak başarıyla tamamlanmıştır. Sistem, hem çevrimdışı (transkripsiyon) hem de çevrimiçi (analiz) modelleri hibrit bir şekilde yöneterek esneklik sağlar.
