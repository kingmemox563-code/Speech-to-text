# PROJE RAPORU: AKILLI SES ANALİZ VE DOĞRULAMA SİSTEMİ

**Hazırlayan: Mehmet Karataş  
**Öğrenci No: 222523002  
**Ders: MÜHENDİSLİKTE BİLGİSAYAR UYGULAMALARI I  

---

## 1. Proje Amacı ve Kapsamı
Bu projenin temel amacı, kullanıcıların sesli konuşmalarını yüksek doğrulukla metne dönüştüren ve bu metinleri gelişmiş yapay zeka algoritmalarıyla analiz eden uçtan uca bir sistem geliştirmektir. Sistem, sadece transkripsiyon yapmakla kalmaz, aynı zamanda konuşmanın duygu durumunu, ana temalarını ve kritik noktalarını belirleyerek profesyonel bir PDF raporu sunar.

## 2. Kullanılan Teknolojiler
Proje, modern Python kütüphaneleri ve yapay zeka modelleri üzerine inşa edilmiştir:
- **Arayüz (Frontend):** `customtkinter` (Modern ve karanlık mod destekli GUI).
- **Ses İşleme:** `sounddevice`, `soundfile`, `FFmpeg`.
- **Transkripsiyon (Speech-to-Text):** `OpenAI Whisper` (Dahili, çevrimdışı çalışabilir GPU/CPU destekli model).
- **Analiz (LLM):** `GPT-4o` (OpenAI API - Metin analizi ve özetleme).
- **Görselleştirme:** `matplotlib` (Duygu analizi grafiği), `wordcloud` (Kelime bulutu).
- **Raporlama:** `fpdf` (Otomatik PDF oluşturma).

## 3. Sistem Gereksinimleri ve Kurulum
Projenin verimli çalışması için donanım tabanlı iki farklı kurulum stratejisi belirlenmiştir:

### 3.1. Donanım Gereksinimleri
- **NVIDIA GPU (Önerilen):** En az 4GB VRAM (CUDA 11.8+ desteğiyle).
- **CPU:** Çok çekirdekli modern bir işlemci (Intel i5/AMD Ryzen 5 ve üzeri).
- **RAM:** En az 8GB.

### 3.2. Model Tavsiyeleri
| Donanım | Önerilen Whisper Modeli | Performans |
| :--- | :--- | :--- |
| **NVIDIA RTX 3060+** | `medium` veya `large` | Çok Yüksek / Hızlı |
| **NVIDIA GTX 1650/1050** | `small` | Orta / Hızlı |
| **Sadece CPU (İyi)** | `base` | Orta / Yavaş |
| **Zayıf Sistemler** | `tiny` | Düşük / Hızlı |

### 3.3. Yazılım Kurulumu
1. **FFmpeg:** Ses işleme için sistemde yüklü ve PATH'e ekli olmalıdır.
2. **PyTorch (CUDA):** `pip install torch --index-url https://download.pytorch.org/whl/cu118`
3. **Kütüphaneler:** `pip install -r requirements.txt`

## 4. Proje İş Paketleri ve Geliştirme Süreci
1. **İş Paketi 1 (GUI):** Kullanıcı dostu, mikrofon seçimi ve ayarların yapılabildiği ana ekran tasarımı.
2. **İş Paketi 2 (Ses & Transkripsiyon):** Gerçek zamanlı ses kaydı ve Whisper modellerinin entegrasyonu.
3. **İş Paketi 3 (Analiz Motoru):** GPT-4o ile metin analizi, duygu skorlama ve WordCloud oluşturma.
4. **İş Paketi 4 (Raporlama):** Tüm verilerin profesyonel bir rapor formatında PDF'e aktarılması.

## 5. Teknik Zorluklar ve Çözümler
- **Karakter Kodlama:** Türkçe karakterlerin (`ığüşöç`) PDF ve konsol çıktılarında hata vermemesi için sistem `UTF-8` ve `latin-1` dönüşümleriyle stabilize edildi.
- **Donanım Uyumluluğu:** Cihazda GPU olup olmadığı otomatik kontrol edilerek sistemin çökmesi engellendi (`torch.cuda.is_available()`).

---
> [!IMPORTANT]
> Proje, bitirme kriterlerini tam olarak karşılamakta olup kod yapısı modüler ve geliştirilebilir şekilde tasarlanmıştır.
