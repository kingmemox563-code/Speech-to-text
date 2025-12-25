"""
analytics.py - Veri Analitiği ve Görselleştirme Modülü
Bu modül, metin verilerinden kelime bulutu (wordcloud) ve duygu analizi grafiklerini 
(pie chart) oluşturmak için kullanılır.
"""

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

class AnalyticsGenerator:
    """
    Metin analizi sonuçlarını görselleştiren yardımcı sınıf.
    Kelime bulutu ve duygu durum grafiklerini üretir.
    """
    def __init__(self):
        # Genişletilmiş Türkçe stopwords (gereksiz kelimeler) listesi
        # Bu kelimeler analiz sırasında dikkate alınmaz.
        self.stopwords = set([
            "ve", "ile", "bir", "bu", "şu", "o", "da", "de", "için", "gibi", "kadar", 
            "olarak", "olan", "ne", "var", "yok", "ama", "fakat", "lakin", "ancak", 
            "ben", "sen", "biz", "siz", "onlar", "mi", "mu", "mı", "mü", "şey", "çok",
            "daha", "en", "ise", "veya", "ya", "hani", "yani", "diye", "her", "zaman",
            "ki", "bi", "miyim", "misin", "mısın", "bunu", "buna", "burada", "şimdi"
        ])
        
        # GUI çakışmasını önlemek için Matplotlib arka plan işleyicisini (backend) 'Agg' olarak ayarla.
        # Bu, grafiklerin bir pencerede açılmak yerine arka planda oluşturulmasını sağlar.
        plt.switch_backend('Agg')

    def generate_wordcloud(self, text, output_path="temp_wordcloud.png"):
        """
        Verilen metni görsel bir kelime bulutuna dönüştürür.
        
        Args:
            text (str): Analiz edilecek ham metin.
            output_path (str): Oluşturulan görselin kaydedileceği dosya yolu.
            
        Returns:
            str: Başarılı ise görselin mutlak yolu, başarısız ise None.
        """
        try:
            # Metin çok kısaysa işlem yapma
            if not text or len(text.strip()) < 5:
                return None
            
            # Kelime bulutu oluşturucu konfigürasyonu
            wc = WordCloud(
                width=1000, height=600,
                background_color='white',
                max_words=100,
                stopwords=self.stopwords,
                colormap='magma', # Modern renk paleti
                font_step=2,
                prefer_horizontal=0.7
            ).generate(text)
            
            # Görseli dosyaya yaz
            wc.to_file(output_path)
            return os.path.abspath(output_path)
        except Exception as e:
            print(f"WordCloud hatası: {e}")
            return None

    def generate_sentiment_chart(self, positive, negative, neutral, output_path="temp_chart.png"):
        """
        Duygu analizi sonuçlarını (pozitif, negatif, nötr) pasta grafiğine dönüştürür.
        
        Args:
            positive (int): Pozitif duygu yüzdesi/skoru.
            negative (int): Negatif duygu yüzdesi/skoru.
            neutral (int): Nötr duygu yüzdesi/skoru.
            output_path (str): Grafiğin kaydedileceği dosya yolu.
            
        Returns:
            str: Başarılı ise görselin mutlak yolu, başarısız ise None.
        """
        try:
            # Eğer tüm değerler 0 ise varsayılan olarak nötr göster
            if positive == 0 and negative == 0 and neutral == 0:
                neutral = 100

            sizes = [positive, negative, neutral]
            labels = ['Pozitif', 'Negatif', 'Nötr']
            # Grafik renkleri: Yeşil, Kırmızı, Gri
            colors = ['#2ecc71', '#e74c3c', '#95a5a6'] 
            explode = (0.05, 0, 0) # Pozitif dilimi hafifçe dışa çıkar
            
            # Grafik alanı oluştur (DPI 100-150 arası netlik için iyidir)
            plt.figure(figsize=(6, 5), dpi=100) 
            patches, texts, autotexts = plt.pie(
                sizes, 
                explode=explode, 
                labels=labels, 
                colors=colors, 
                autopct='%1.1f%%', 
                startangle=140, 
                shadow=False
            )
            
            # Etiketlerin görünürlüğünü ve yazı tipini ayarla
            for text in texts:
                text.set_color('#2c3e50')
                text.set_weight('bold')
            
            plt.axis('equal') # Dairenin tam yuvarlak olmasını sağlar
            plt.title("Konuşma Duygu Analiz Dağılımı", pad=20, fontdict={'fontsize': 14, 'fontweight': 'bold'})
            
            plt.tight_layout()
            plt.savefig(output_path, transparent=False, dpi=150)
            plt.close() # Hafıza sızıntısını önlemek için kapat
            return os.path.abspath(output_path)
        except Exception as e:
            print(f"Grafik hatası: {e}")
            return None
