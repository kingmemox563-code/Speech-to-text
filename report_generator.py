from fpdf import FPDF
import os
import time

class PDFReport(FPDF):
    def header(self):
        # Logo Kontrolü
        if os.path.exists("logo.png"):
            self.image("logo.png", 10, 8, 25)
            self.set_x(40)
        
        # Başlık Tasarımı
        self.set_font('Arial', 'B', 16)
        self.set_text_color(44, 62, 80) # Lacivert tonu
        # 'Ğ' ve diğerlerini ASCII-safe yapıyoruz
        self.cell(0, 10, 'AKILLI SES ANALIZ VE DOGRULAMA RAPORU', 0, 1, 'C')
        
        self.set_font('Arial', 'I', 10)
        self.set_text_color(127, 140, 141) # Gri tonu
        self.cell(0, 10, f'Rapor Tarihi: {time.strftime("%d/%m/%Y %H:%M")}', 0, 1, 'C')
        self.ln(10)
        # Ayırıcı Çizgi
        self.line(10, 35, 200, 35)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(149, 165, 166)
        # Türkçe karakterleri temizliyoruz: 'ı' -> 'i' vb.
        self.cell(0, 10, f'Sayfa {self.page_no()} | Yapay Zeka Destekli Analiz Sistemi', 0, 0, 'C')

class ReportGenerator:
    def create_report(self, filename, text, analysis_summary, sentiment_stats, visuals_paths):
        """
        Gemini/GPT'den gelen analizleri profesyonel PDF'e dönüştürür.
        """
        # Türkçe karakterleri desteklemek için eşleme (Arial latin-1 uyumu için)
        tr_map = str.maketrans("ığüşöçİĞÜŞÖÇ", "igusocIGUSOC")
        
        pdf = PDFReport()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # 1. YÖNETİCİ ÖZETİ
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(41, 128, 185)
        pdf.cell(0, 10, "1. YONETICI OZETI", 0, 1, 'L')
        
        pdf.set_font("Arial", '', 11)
        pdf.set_text_color(0, 0, 0)
        # Analiz özetini temizleyip yazdır
        clean_summary = analysis_summary.translate(tr_map).encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 7, clean_summary)
        pdf.ln(10)
        
        # 2. DUYGU DURUM ANALİZİ
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(41, 128, 185)
        pdf.cell(0, 10, "2. DUYGU DURUM ANALIZI", 0, 1, 'L')
        
        # Pasta Grafiği Ekleme
        if visuals_paths.get("chart") and os.path.exists(visuals_paths["chart"]):
            # Grafiği ortala
            pdf.image(visuals_paths["chart"], x=55, w=100)
            pdf.ln(5)
            
        pdf.set_font("Arial", '', 11)
        pdf.set_text_color(44, 62, 80)
        stats_text = (f"Pozitiflik: %{sentiment_stats.get('pos', 0)} | "
                      f"Negatiflik: %{sentiment_stats.get('neg', 0)} | "
                      f"Notr: %{sentiment_stats.get('neu', 0)}")
        pdf.cell(0, 10, stats_text, 0, 1, 'C')
        pdf.ln(5)

        # 3. KELİME BULUTU
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(41, 128, 185)
        pdf.cell(0, 10, "3. ANAHTAR KELIME ANALIZI (Word Cloud)", 0, 1, 'L')
        
        if visuals_paths.get("wordcloud") and os.path.exists(visuals_paths["wordcloud"]):
            pdf.image(visuals_paths["wordcloud"], x=15, w=180)
            pdf.ln(10)
            
        # 4. TRANSKRIPSIYON (TAM METİN)
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(41, 128, 185)
        pdf.cell(0, 10, "4. KONUSMA DOKUMU (Dogrulanmis Metin)", 0, 1, 'L')
        
        pdf.set_font("Arial", '', 9)
        pdf.set_text_color(0, 0, 0)
        
        # Metni temizleyip yazdır
        safe_text = text.translate(tr_map).encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 5, safe_text)
            
        try:
            pdf.output(filename)
            return filename
        except Exception as e:
            print(f"PDF Kayıt Hatası: {e}")
            return None
