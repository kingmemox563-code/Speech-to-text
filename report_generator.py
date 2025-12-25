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
        # Türkçe karakterleri destekleyen fontu yükle (Windows standart dizininden)
        font_path = r"C:\Windows\Fonts\arial.ttf"
        font_bold_path = r"C:\Windows\Fonts\arialbd.ttf"
        font_italic_path = r"C:\Windows\Fonts\ariali.ttf"
        
        if os.path.exists(font_path):
            self.add_font("ArialTR", "", font_path)
        if os.path.exists(font_bold_path):
            self.add_font("ArialTR", "B", font_bold_path)
        if os.path.exists(font_italic_path):
            self.add_font("ArialTR", "I", font_italic_path)

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
