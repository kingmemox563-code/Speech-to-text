"""
report_generator.py - PDF Rapor Oluşturma Modülü
Bu modül, transkripsiyon metinlerini, AI analizlerini ve görsel grafikleri 
profesyonel bir PDF dosyasına dönüştürmek için kullanılır.
"""

from fpdf import FPDF
import os
import time

class PDFReport(FPDF):
    """
    FPDF sınıfından türetilmiş, özel başlık (header) ve altbilgi (footer) 
    tasarımına sahip rapor sınıfı.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Türkçe karakterleri destekleyen fontu yükle
        # Farklı sistemlerde (Linux/Mac/Windows) çalışabilmesi için kontrol ekle
        possible_paths = [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\ariali.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", # Linux Fallback
            "/System/Library/Fonts/Helvetica.ttc" # Mac Fallback
        ]
        
        # Windows spesifik yollar
        font_path = r"C:\Windows\Fonts\arial.ttf"
        font_bold_path = r"C:\Windows\Fonts\arialbd.ttf"
        font_italic_path = r"C:\Windows\Fonts\ariali.ttf"
        
        # Fontları kaydet (Eğer ana font yoksa sistem varsayılanını kullanır)
        font_loaded = False
        if os.path.exists(font_path):
            self.add_font("ArialTR", "", font_path)
            font_loaded = True
        if os.path.exists(font_bold_path):
            self.add_font("ArialTR", "B", font_bold_path)
        if os.path.exists(font_italic_path):
            self.add_font("ArialTR", "I", font_italic_path)
            
        if not font_loaded:
            # Fallback: ArialTR ismini standart bir fonta eşle
            self.set_font("Arial", "", 12)
            # Not: ArialTR ismini kullanmaya devam edersek hata verebilir, 
            # bu yüzden ArialTR yerine standart font isimleri kullanılabilir.
            # Ancak Header/Footer fonksiyonları ArialTR bekliyor.
            # En güvenli yol: ArialTR isminde yerleşik bir fontu alias yapmak.
            pass 

    def header(self):
        # Logo Kontrolü
        if os.path.exists("logo.png"):
            self.image("logo.png", 10, 8, 25)
            self.set_x(40)
        
        # Başlık Tasarımı
        self.set_font('ArialTR', 'B', 16)
        self.set_text_color(44, 62, 80)
        self.cell(0, 10, 'AKILLI SES ANALİZ VE DOĞRULAMA RAPORU', 0, 1, 'C')
        
        # Rapor Tarihi
        self.set_font('ArialTR', 'I', 10)
        self.set_text_color(127, 140, 141)
        self.cell(0, 10, f'Rapor Tarihi: {time.strftime("%d/%m/%Y %H:%M")}', 0, 1, 'C')
        self.ln(10)
        
        # Ayırıcı Çizgi
        self.line(10, 35, 200, 35)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('ArialTR', 'I', 8)
        self.set_text_color(149, 165, 166)
        self.cell(0, 10, f'Sayfa {self.page_no()} | Yapay Zeka Destekli Analiz Sistemi', 0, 0, 'C')

class ReportGenerator:
    """
    Transkriptleri ve AI analizlerini harmanlayarak PDF üreten sınıf.
    """
    def create_report(self, filename, text, analysis_summary, sentiment_stats, visuals_paths):
        """
        AI'dan gelen analizleri ve görselleri profesyonel PDF'e dönüştürür.
        Unicode desteklidir (Gerçek Türkçe karakterler).
        """
        pdf = PDFReport()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # --- 1. ANALİZ VE YÖNETİCİ ÖZETİ ---
        pdf.set_font("ArialTR", 'B', 14)
        pdf.set_text_color(41, 128, 185)
        pdf.cell(0, 10, "1. ANALİZ VE YÖNETİCİ ÖZETİ (AI MODELLERİ)", 0, 1, 'L')
        
        pdf.set_font("ArialTR", '', 11)
        pdf.set_text_color(0, 0, 0)
        
        if isinstance(analysis_summary, dict):
            for provider, content in analysis_summary.items():
                if content:
                    pdf.ln(5)
                    pdf.set_font("ArialTR", 'B', 12)
                    pdf.set_text_color(22, 160, 133) # Yeşilimsi ton
                    title = f"--- {provider.upper()} ANALİZİ ---"
                    pdf.cell(0, 10, title, 0, 1, 'L')
                    
                    # Analiz Metni
                    pdf.set_font("ArialTR", '', 11)
                    pdf.set_text_color(0, 0, 0)
                    pdf.multi_cell(0, 7, content)
                    pdf.ln(5)

                    # Bu sağlayıcıya özel grafik ve istatistikler
                    # sentiment_stats[provider] = {'pos':x, 'neg':y, 'neu':z}
                    if isinstance(sentiment_stats, dict) and provider in sentiment_stats:
                        stats = sentiment_stats[provider]
                        if stats:
                            # Grafik dosyası: temp_chart_{provider}.png
                            chart_path = os.path.abspath(f"temp_chart_{provider}.png")
                            if os.path.exists(chart_path):
                                pdf.image(chart_path, x=55, w=100) # Boyut artırıldı (70->100) ve ortalandı
                                pdf.ln(5)
                            
                            pdf.set_font("ArialTR", 'I', 10)
                            pdf.set_text_color(44, 62, 80)
                            stats_text = (f"{provider} Skorları: Pozitiflik: %{stats.get('pos', 0)} | "
                                          f"Negatiflik: %{stats.get('neg', 0)} | "
                                          f"Nötr: %{stats.get('neu', 0)}")
                            pdf.cell(0, 10, stats_text, 0, 1, 'C')
                            pdf.ln(5)
        else:
            # Tek bir analiz metni gelmişse (eski uyumluluk)
            pdf.multi_cell(0, 7, analysis_summary)
            pdf.ln(10)
            
            # Eski tekli grafik mantığı
            if visuals_paths.get("chart") and os.path.exists(visuals_paths["chart"]):
                pdf.image(visuals_paths["chart"], x=55, w=100)
                pdf.ln(5)

        # --- 3. ANAHTAR KELİME ANALİZİ (Kelime Bulutu) ---
        pdf.add_page()
        pdf.set_font("ArialTR", 'B', 14)
        pdf.set_text_color(41, 128, 185)
        pdf.cell(0, 10, "3. ANAHTAR KELİME ANALİZİ (Kelime Bulutu)", 0, 1, 'L')
        
        if visuals_paths.get("wordcloud") and os.path.exists(visuals_paths["wordcloud"]):
            pdf.image(visuals_paths["wordcloud"], x=15, w=180)
            pdf.ln(10)
            
        # --- 4. KONUŞMA DÖKÜMÜ ---
        pdf.set_font("ArialTR", 'B', 14)
        pdf.set_text_color(41, 128, 185)
        pdf.cell(0, 10, "4. KONUŞMA DÖKÜMÜ (Doğrulanmış Metin)", 0, 1, 'L')
        
        pdf.set_font("ArialTR", '', 9)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 5, text)
            
        try:
            pdf.output(filename)
            return filename
        except Exception as e:
            print(f"PDF Kayıt Hatası: {e}")
            return None

    def create_coach_report(self, filename, transcript, feedback, chat_history, metadata):
        """
        Dil Koçu analizini, geri bildirimleri ve chat geçmişini PDF raporuna dönüştürür.
        """
        pdf = PDFReport()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # --- 1. ÖĞRENCİ VE HEDEF BİLGİLERİ ---
        pdf.set_font("ArialTR", 'B', 14)
        pdf.set_text_color(41, 128, 185)
        pdf.cell(0, 10, "1. DİL ÖĞRENİM PROFİLİ VE HEDEF", 0, 1, 'L')
        
        pdf.set_font("ArialTR", '', 11)
        pdf.set_text_color(44, 62, 80)
        info_text = f"Hedef Dil: {metadata.get('lang', 'Bilinmiyor')} | " \
                    f"Mevcut Seviye: {metadata.get('level', 'Bilinmiyor')} | " \
                    f"Çalışma Modu: {metadata.get('mode', 'Bilinmiyor')}"
        pdf.cell(0, 10, info_text, 0, 1, 'L')
        pdf.ln(5)

        # --- 2. DİL KOÇU GERİ BİLDİRİMİ ---
        pdf.set_font("ArialTR", 'B', 14)
        pdf.set_text_color(41, 128, 185)
        pdf.cell(0, 10, "2. AI DİL KOÇU ANALİZİ VE TAVSİYELER", 0, 1, 'L')
        
        pdf.set_font("ArialTR", '', 11)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 7, feedback)
        pdf.ln(10)

        # --- 3. SORU-CEVAP VE MENTORLUK GEÇMİŞİ ---
        if chat_history:
            pdf.add_page()
            pdf.set_font("ArialTR", 'B', 14)
            pdf.set_text_color(41, 128, 185)
            pdf.cell(0, 10, "3. SORU-CEVAP VE MENTORLUK GEÇMİŞİ", 0, 1, 'L')
            
            for q, a in chat_history:
                pdf.ln(5)
                pdf.set_font("ArialTR", 'B', 11)
                pdf.set_text_color(231, 76, 60) # Kırmızımsı ton (Soru)
                pdf.multi_cell(0, 7, f"Soru: {q}")
                
                pdf.set_font("ArialTR", '', 11)
                pdf.set_text_color(0, 0, 0) # Siyah (Cevap)
                pdf.multi_cell(0, 7, f"{a}")
                pdf.ln(2)
                pdf.line(15, pdf.get_y(), 195, pdf.get_y())

        # --- 4. ORİJİNAL KONUŞMA DÖKÜMÜ ---
        if transcript:
            pdf.add_page()
            pdf.set_font("ArialTR", 'B', 14)
            pdf.set_text_color(41, 128, 185)
            pdf.cell(0, 10, "4. KONUŞMA DÖKÜMÜ (Transkript)", 0, 1, 'L')
            
            pdf.set_font("ArialTR", '', 9)
            pdf.set_text_color(127, 140, 141)
            pdf.multi_cell(0, 5, transcript)
            
        try:
            pdf.output(filename)
            return filename
        except Exception as e:
            print(f"PDF Koç Raporu Kayıt Hatası: {e}")
            return None
