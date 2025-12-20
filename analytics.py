import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

class AnalyticsGenerator:
    def __init__(self):
        # Genişletilmiş Türkçe stopwords listesi
        self.stopwords = set([
            "ve", "ile", "bir", "bu", "şu", "o", "da", "de", "için", "gibi", "kadar", 
            "olarak", "olan", "ne", "var", "yok", "ama", "fakat", "lakin", "ancak", 
            "ben", "sen", "biz", "siz", "onlar", "mi", "mu", "mı", "mü", "şey", "çok",
            "daha", "en", "ise", "veya", "ya", "hani", "yani", "diye", "her", "zaman",
            "ki", "bi", "miyim", "misin", "mısın", "bunu", "buna", "burada", "şimdi"
        ])
        
        # GUI çakışmasını önlemek için arka plan işleyiciyi ayarla
        plt.switch_backend('Agg')

    def generate_wordcloud(self, text, output_path="temp_wordcloud.png"):
        """Metni görsel kelime bulutuna dönüştürür."""
        try:
            if not text or len(text.strip()) < 5:
                return None
            
            # Kelime bulutu tasarımı (Görsel kalite artırıldı)
            wc = WordCloud(
                width=1000, height=600,
                background_color='white',
                max_words=100,
                stopwords=self.stopwords,
                colormap='magma', # Daha modern bir renk paleti
                font_step=2,
                prefer_horizontal=0.7
            ).generate(text)
            
            wc.to_file(output_path)
            return os.path.abspath(output_path)
        except Exception as e:
            print(f"WordCloud hatası: {e}")
            return None

    def generate_sentiment_chart(self, positive, negative, neutral, output_path="temp_chart.png"):
        """Duygu dağılımını yüksek kaliteli bir pasta grafiğine dönüştürür."""
        try:
            # Eğer tüm değerler 0 ise grafik çizme
            if positive == 0 and negative == 0 and neutral == 0:
                neutral = 100 # Varsayılan nötr

            sizes = [positive, negative, neutral]
            labels = ['Pozitif', 'Negatif', 'Nötr']
            # Modern ve profesyonel renkler
            colors = ['#2ecc71', '#e74c3c', '#95a5a6'] 
            explode = (0.05, 0, 0) # Sadece pozitif kısmı hafif öne çıkar
            
            plt.figure(figsize=(6, 5), dpi=100) # DPI artırılarak netlik sağlandı
            patches, texts, autotexts = plt.pie(
                sizes, 
                explode=explode, 
                labels=labels, 
                colors=colors, 
                autopct='%1.1f%%', 
                startangle=140, 
                shadow=False # Daha temiz görünüm için gölge kaldırıldı
            )
            
            # Etiket fontlarını düzelt (Türkçe karakter uyumu için)
            for text in texts:
                text.set_color('#2c3e50')
                text.set_weight('bold')
            
            plt.axis('equal')
            plt.title("Konuşma Duygu Analiz Dağılımı", pad=20, fontdict={'fontsize': 14, 'fontweight': 'bold'})
            
            plt.tight_layout()
            plt.savefig(output_path, transparent=False, dpi=150)
            plt.close()
            return os.path.abspath(output_path)
        except Exception as e:
            print(f"Grafik hatası: {e}")
            return None
