"""
gui.py - Ana Kullanıcı Arayüzü (GUI) Modülü
Bu modül, uygulamanın görsel arayüzünü (CustomTkinter), ses kayıt kontrollerini, 
API entegrasyonlarını ve raporlama özelliklerini bir araya getirir.
"""

import customtkinter as ctk
from tkinter import messagebox, filedialog
import threading
import sounddevice as sd
import soundfile as sf
import json
import torch
import os
import numpy as np
from openai import OpenAI
from fpdf import FPDF
from gemini_client import GeminiClient
from dotenv import load_dotenv, set_key
import datetime

# .env dosyasını yükle (API anahtarları için)
load_dotenv()

# Karakter hatalarını önlemek için sistem dilini UTF-8 yapıyoruz
os.environ["PYTHONIOENCODING"] = "utf-8"

# Dinamik modül yüklemeleri (Opsiyonel bileşenler)
try:
    from analytics import AnalyticsGenerator
    from report_generator import ReportGenerator
    from visualizer import AudioVisualizer
except ImportError:
    # Eğer bu dosyalar mevcut değilse uygulama hatasız çalışmaya devam eder
    AnalyticsGenerator = None
    ReportGenerator = None
    AudioVisualizer = None

class App(ctk.CTk):
    """
    Uygulamanın ana penceresini ve tüm mantıksal akışını yöneten sınıf.
    """
    def __init__(self):
        super().__init__()
        
        # Donanım ve Durum Ayarları
        # Eğer NVIDIA GPU (CUDA) varsa kullan, yoksa CPU kullan
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_recording = False
        self.audio_frames = [] # Kayıt sırasında ses verilerinin toplandığı liste
        self.api_key = "" # OpenAI key
        self.fs = 16000 # Whisper için standart örnekleme hızı (Sample Rate)
        self.selected_mic_index = self.get_default_mic()
        
        # Son Analiz ve Transkript Verileri
        self.last_analysis = ""
        self.last_transcript = ""
        self.sentiment_stats = {'pos': 33, 'neg': 33, 'neu': 34}
        self.gemini_api_key = ""
        
        # Ana Pencere Konfigürasyonu
        self.title("Ses Analiz Sistemi")
        self.geometry("1300x950")
        ctk.set_appearance_mode("dark") # Koyu tema varsayılan
        self.protocol("WM_DELETE_WINDOW", self.on_app_closing)
        
        # Arayüzü oluştur ve kayıtlı anahtarları yükle
        self.setup_ui()
        self.load_api_key()
        
        # Analiz Sonuçlarını Saklama (Çoklu PDF raporu için)
        self.analysis_results = {"OpenAI": "", "Gemini": ""}
        self.all_sentiment_stats = {"OpenAI": None, "Gemini": None}
        
    def get_default_mic(self):
        """Sistemdeki varsayılan mikrofonun indeksini bulur."""
        try:
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0:
                    return i
        except:
            return None
        return None

    def get_mic_list(self):
        """Kullanılabilir mikrofonların listesini döner."""
        try:
            devices = sd.query_devices()
            return [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        except:
            return ["Mikrofon Bulunamadı"]

    def setup_ui(self):
        """Tüm görsel bileşenleri (butonlar, paneller, textboxlar) oluşturur."""
        # Kayıtlar klasörü yoksa oluştur
        if not os.path.exists("recordings"):
            os.makedirs("recordings")
            
        # Grid (Izgara) sistemini yapılandır
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- SOL PANEL (SideBar - Ayarlar) ---
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="AYARLAR", font=("Arial", 20, "bold")).pack(pady=20)
        
        # Donanım Bilgisi Gösterimi (CUDA varsa turkuaz, yoksa turuncu)
        color = "#00adb5" if self.device == "cuda" else "orange"
        ctk.CTkLabel(self.sidebar, text=f"Donanım: {self.device.upper()}", text_color=color).pack()

        # Whisper Model Seçimi
        ctk.CTkLabel(self.sidebar, text="Whisper Modeli:").pack(pady=(10,0))
        self.model_combo = ctk.CTkComboBox(self.sidebar, values=["tiny", "base", "small", "medium"])
        self.model_combo.set("medium")
        self.model_combo.pack(pady=5)

        # Kaynak Dil Seçimi
        ctk.CTkLabel(self.sidebar, text="Kaynak Dil:").pack()
        self.lang_combo = ctk.CTkComboBox(self.sidebar, values=["turkish", "english", "german", "french", "spanish", "italian", "russian"])
        self.lang_combo.set("turkish")
        self.lang_combo.pack(pady=5)

        # Mikrofon Seçimi Combo Box
        ctk.CTkLabel(self.sidebar, text="Mikrofon Seçin:").pack(pady=(10,0))
        self.mic_combo = ctk.CTkComboBox(self.sidebar, values=self.get_mic_list(), command=self.change_mic)
        if self.selected_mic_index is not None:
            mics = self.get_mic_list()
            for m in mics:
                if m.startswith(f"{self.selected_mic_index}:"):
                    self.mic_combo.set(m)
                    break
        self.mic_combo.pack(pady=5)

        # İngilizceye Çeviri Seçeneği
        self.translate_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(self.sidebar, text="İngilizceye Çevir", variable=self.translate_var).pack(pady=10)

        # Kayıt Başlat/Durdur Butonu
        self.record_btn = ctk.CTkButton(self.sidebar, text="KAYDI BAŞLAT", fg_color="green", command=self.toggle_recording)
        self.record_btn.pack(pady=10, padx=20)

        # Dosyadan Ses Yükleme Butonu
        self.file_btn = ctk.CTkButton(self.sidebar, text="SES DOSYASI SEÇ", fg_color="#34495e", command=self.process_audio_file)
        self.file_btn.pack(pady=10, padx=20)

        # Otomatik Kayıt Switch'i
        self.autosave_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(self.sidebar, text="Sesi Otomatik Kaydet", variable=self.autosave_var).pack(pady=10)

        # Klasör ve PDF İşlemleri
        self.open_folder_btn = ctk.CTkButton(self.sidebar, text="KAYITLARI AÇ", fg_color="#7f8c8d", command=self.open_recordings_folder)
        self.open_folder_btn.pack(pady=10, padx=20)

        self.pdf_btn = ctk.CTkButton(self.sidebar, text="PDF OLARAK KAYDET", fg_color="#e67e22", command=self.save_as_pdf)
        self.pdf_btn.pack(pady=10, padx=20)

        # API Anahtarları Giriş Alanları
        ctk.CTkLabel(self.sidebar, text="OpenAI API Anahtarı:").pack(pady=(10, 0))
        self.api_entry = ctk.CTkEntry(self.sidebar, placeholder_text="OpenAI API Key", show="*")
        self.api_entry.pack(pady=(5, 5), padx=20)
        
        ctk.CTkLabel(self.sidebar, text="Gemini API Anahtarı:").pack(pady=(5, 0))
        self.gemini_api_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Gemini API Key", show="*")
        self.gemini_api_entry.pack(pady=(5, 5), padx=20)
        
        ctk.CTkButton(self.sidebar, text="Anahtarları Kaydet", command=self.save_api_keys).pack(pady=5)

        # --- SAĞ ANALİZ PANELİ ---
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=20, pady=10)

        # Sistem Durum Etiketi
        self.status_label = ctk.CTkLabel(self.main_container, text="Sistem Hazır", text_color="#00adb5")
        self.status_label.pack(anchor="w")

        # Canlı Ses Görselleştirici (Mavi Dalga Barı)
        if AudioVisualizer:
            self.visualizer = AudioVisualizer(self.main_container)
            self.visualizer.pack(fill="x", pady=10)

        # Transkriptlerin ve Analizlerin Göründüğü Ana Metin Kutusu
        self.textbox = ctk.CTkTextbox(self.main_container, font=("Consolas", 15))
        self.textbox.pack(fill="both", expand=True, pady=(0, 10))

        # Analiz Başlatma Butonları
        self.analyze_btn = ctk.CTkButton(self.main_container, text="GPT-4o İLE ANALİZ ET", fg_color="#10a37f", height=50, command=self.run_analysis)
        self.analyze_btn.pack(fill="x", pady=5)

        self.gemini_analyze_btn = ctk.CTkButton(self.main_container, text="GEMINI İLE ANALİZ ET", fg_color="#4285f4", height=50, command=self.run_gemini_analysis)
        self.gemini_analyze_btn.pack(fill="x", pady=5)

    def change_mic(self, value):
        """Kullanıcının seçtiği mikrofon indeksini günceller."""
        try:
            self.selected_mic_index = int(value.split(":")[0])
        except:
            pass

    # --- SES KAYIT VE İŞLEME MANTIĞI ---
    def toggle_recording(self):
        """Kayıt düğmesine basıldığında başlatma/durdurma işlemini yapar."""
        if not self.is_recording:
            self.is_recording = True
            self.record_btn.configure(text="KAYDI DURDUR", fg_color="red")
            self.status_label.configure(text="Kaydediliyor...", text_color="red")
            self.audio_frames = []
            # Çakışmayı önlemek için kayıt işlemini ayrı bir thread'de başlat
            threading.Thread(target=self._record_thread, daemon=True).start()
        else:
            self.is_recording = False
            self.record_btn.configure(text="KAYDI BAŞLAT", fg_color="green")
            self.status_label.configure(text="Kayıt durduruldu.", text_color="#00adb5")
            if hasattr(self, 'visualizer'):
                self.visualizer.clear()

    def _record_thread(self):
        """Mikrofondan ham ses verilerini okuyan iş parçacığı."""
        try:
            with sd.InputStream(samplerate=self.fs, channels=1, callback=self._audio_callback, device=self.selected_mic_index):
                while self.is_recording:
                    sd.sleep(100)
        except Exception as e:
            self.is_recording = False
            self.after(0, lambda: messagebox.showerror("Donanım Hatası", f"Mikrofon hatası: {e}"))
            return
        
        # Kayıt durduğunda veriyi birleştir ve geçici dosyaya yaz
        audio_path = "temp_recording.wav"
        audio_data = np.concatenate(self.audio_frames, axis=0)
        sf.write(audio_path, audio_data, self.fs)

        # Eğer otomatik kayıt açıksa recordings klasörüne tarih-saat ile kaydet
        if self.autosave_var.get():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join("recordings", f"kayit_{timestamp}.wav")
            sf.write(save_path, audio_data, self.fs)
            print(f"Ses kaydedildi: {save_path}")

        # Transkripsiyon sürecini başlat
        self._transcribe_file(audio_path)

    def _audio_callback(self, indata, frames, time, status):
        """Mikrofondan gelen her ses paketinde tetiklenir."""
        if self.is_recording:
            self.audio_frames.append(indata.copy())
            # Görselleştiriciyi güncelle
            if hasattr(self, 'visualizer'):
                self.visualizer.update_visuals(indata)

    def _transcribe_file(self, path):
        """Ses dosyasını Whisper kullanarak metne dönüştürür."""
        try:
            from transcriber import Transcriber
            self.status_label.configure(text="Metne dönüştürülüyor...")
            
            task = "translate" if self.translate_var.get() else "transcribe"
            model_type = self.model_combo.get()
            
            # Whisper transkripsiyon işlemi
            ts = Transcriber(device=self.device, model_type=model_type)
            res = ts.model.transcribe(path, language=self.lang_combo.get(), task=task)
            
            # Karakter hatalarını önlemek için temizlik yap
            clean_text = res['text'].encode('utf-8', 'replace').decode('utf-8')
            self.last_transcript = clean_text 
            
            # Sonucu ana metodun thread'inde (Main UI Thread) metin kutusuna ekle
            self.after(0, lambda: self.textbox.insert("end", f"\n[TRANSKRIPT]:\n{clean_text}\n"))
            self.status_label.configure(text="İşlem tamamlandı.")
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Hata", f"Transkripsiyon Hatası: {e}"))

    def process_audio_file(self):
        """Bilgisayardan bir ses dosyası seçilmesini sağlar."""
        path = filedialog.askopenfilename(filetypes=[("Ses Dosyası", "*.wav *.mp3 *.m4a")])
        if path:
            threading.Thread(target=lambda: self._transcribe_file(path), daemon=True).start()

    def open_recordings_folder(self):
        """Kayıtların tutulduğu klasörü Windows Explorer'da açar."""
        folder_path = os.path.abspath("recordings")
        if os.path.exists(folder_path):
            os.startfile(folder_path)
        else:
            messagebox.showerror("Hata", "Kayıtlar klasörü bulunamadı!")

    # --- GPT-4o ANALİZ METOTLARI ---
    def run_analysis(self):
        """Metin kutusundaki verileri GPT-4o ile analiz etmek üzere gönderir."""
        text = self.textbox.get("1.0", "end").strip()
        if text:
            threading.Thread(target=self._gpt_logic, args=(text,), daemon=True).start()

    def _gpt_logic(self, text):
        """Arka planda OpenAI API isteğini yönetir."""
        try:
            if not self.api_key:
                self.after(0, lambda: messagebox.showwarning("Anahtar Eksik", "Lütfen OpenAI API anahtarınızı kaydedin."))
                return

            safe_text = text.encode('utf-8', 'replace').decode('utf-8')
            client = OpenAI(api_key=self.api_key)
            self.status_label.configure(text="GPT-4o analiz ediyor...")
            
            prompt = self._get_analysis_prompt(safe_text)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Transkriptleri profesyonelce ve detaylıca Türkçe analiz et. Her segmenti ayrı ayrı ve bütünü toplu analiz et."},
                    {"role": "user", "content": prompt}
                ]
            )
            analysis = response.choices[0].message.content
            self._process_analysis_result(analysis, safe_text, "OpenAI")
        except Exception as e:
            err = str(e).encode('utf-8', 'ignore').decode('utf-8')
            self.after(0, lambda: messagebox.showerror("API Hatası", f"Hata: {err}"))

    # --- GEMINI ANALİZ METOTLARI ---
    def run_gemini_analysis(self):
        """Metin kutusundaki verileri Google Gemini ile analiz eder."""
        text = self.textbox.get("1.0", "end").strip()
        if text:
            threading.Thread(target=self._gemini_logic, args=(text,), daemon=True).start()

    def _gemini_logic(self, text):
        """Arka planda Gemini API isteğini yönetir."""
        try:
            if not self.gemini_api_key:
                self.after(0, lambda: messagebox.showwarning("Anahtar Eksik", "Lütfen Gemini API anahtarınızı kaydedin."))
                return

            safe_text = text.encode('utf-8', 'replace').decode('utf-8')
            client = GeminiClient(api_key=self.gemini_api_key)
            self.status_label.configure(text="Gemini analiz ediyor...")
            
            prompt = self._get_analysis_prompt(safe_text)
            system_instruction = "Transkriptleri profesyonelce ve detaylıca Türkçe analiz et. Her segmenti ayrı ayrı ve bütünü toplu analiz et."
            
            analysis = client.generate_content(prompt, system_instruction=system_instruction)
            self._process_analysis_result(analysis, safe_text, "Gemini")
        except Exception as e:
            err = str(e).encode('utf-8', 'ignore').decode('utf-8')
            self.after(0, lambda: messagebox.showerror("API Hatası", f"Hata: {err}"))

    def _get_analysis_prompt(self, safe_text):
        """AI modellerine gönderilecek kapsamlı analiz komutunu (prompt) döner."""
        return f"""
        Sen profesyonel bir veri analisti ve dil bilimcisin. 
        Aşağıdaki metni bir bitirme projesi raporu ciddiyetinde ve derinliğinde analiz et.
        
        BAĞLAM VE GÖREV:
        Bu metin bir veya birden fazla ses kaydının transkriptini içerebilir. Her bir kayıt '[TRANSCRIPT]' başlığı ile belirtilmiştir.
        Senin görevin:
        1. Ekranda kaç farklı veri/kayıt varsa her birini önce KENDİ İÇİNDE analiz et (Tema, Duygu, Önemli Noktalar).
        2. Ardından tüm kayıtları BÜTÜNCÜL olarak ele alıp aralarındaki kontrastı sentezle.
        
        Lütfen şu başlıklar altında ÇOK DETAYLI bir rapor sun:
        1. TRANSKRİPT BAZLI ANALİZ
        2. GENEL ÖZET VE SENTEZ
        3. TEMEL KONULAR
        4. DERİN DUYGU ANALİZİ
        5. KRİTİK NOKTALAR VE EYLEM PLANI
        6. ÖNERİLER
        
        [SKORLAR]:
        (Tüm metnin ağırlıklı duygusunu yansıtan tek bir skor seti. ÖNEMLİ: Bu üç değerin TOPLAMI TAM 100 OLMALIDIR!)
        POZİTİF: (sayı)
        NEGATİF: (sayı)
        NÖTR: (sayı)
        
        METİN: {safe_text}
        """

    def _process_analysis_result(self, analysis, safe_text, provider):
        """AI'dan gelen analiz sonucunu işler ve görselleri üretir."""
        if AnalyticsGenerator:
            try:
                analyzer = AnalyticsGenerator()
                # Kelime bulutu oluştur
                analyzer.generate_wordcloud(safe_text)
                
                # AI yanıtından skorları ayıkla
                pos, neg, neu = 33, 33, 34 
                for line in analysis.split('\n'):
                    if "POZİTİF:" in line: pos = int(''.join(filter(str.isdigit, line)) or 33)
                    if "NEGATİF:" in line: neg = int(''.join(filter(str.isdigit, line)) or 33)
                    if "NÖTR:" in line: neu = int(''.join(filter(str.isdigit, line)) or 34)

                # --- NORMALİZASYON (Toplamı 100'e sabitleme) ---
                total = pos + neg + neu
                if total > 0:
                    pos = round((pos / total) * 100)
                    neg = round((neg / total) * 100)
                    neu = 100 - (pos + neg) # Kalanı nötre vererek toplamı tam 100 yap
                # -----------------------------------------------

                # Sağlayıcı ismini normalize et (OpenAI vs Gemini)
                provider_key = "OpenAI" if "OpenAI" in provider else "Gemini"
                
                # Sağlayıcıya özel istatistikleri ve grafiği sakla
                stats = {'pos': pos, 'neg': neg, 'neu': neu}
                self.all_sentiment_stats[provider_key] = stats
                self.sentiment_stats = stats # Son yapılan analiz (eski uyumluluk)
                
                # Pasta grafiği oluştur (Sağlayıcıya özel dosya adı)
                chart_path = f"temp_chart_{provider_key}.png"
                analyzer.generate_sentiment_chart(pos, neg, neu, output_path=chart_path)
                
                # Standart isimle de kaydet (eski uyumluluk/tekli mod için)
                analyzer.generate_sentiment_chart(pos, neg, neu, output_path="temp_chart_Analiz.png")
            except Exception as ae:
                print(f"Görsel Analiz Hatası: {ae}")

        self.last_analysis = analysis 
        self.analysis_results[provider] = analysis # Çoklu analiz için sakla
        self.after(0, lambda: self.textbox.insert("end", f"\n\n[ANALİZ ({provider})]:\n{analysis}\n"))
        self.status_label.configure(text=f"Analiz {provider} kullanılarak tamamlandı.")

    # --- PDF VE RAPORLAMA ---
    def save_as_pdf(self):
        """Analiz sonuçlarını ve görselleri profesyonel bir PDF raporuna dönüştürür."""
        text = self.textbox.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Uyarı", "Metin kutusu boş!")
            return
        
        transcript = self.last_transcript if self.last_transcript else text
        
        # Eğer hem OpenAI hem Gemini analizi varsa ikisini de gönder
        # Her zaman sözlük yapısında göndererek grafik karmaşasını önle
        active_analyses = {k: v for k, v in self.analysis_results.items() if v}
        if not active_analyses and self.last_analysis:
            # Sadece tek bir analiz varsa (eski sistemden kalan)
            active_analyses = {"Analiz": self.last_analysis}
            report_stats = {"Analiz": self.sentiment_stats}
        else:
            report_stats = self.all_sentiment_stats

        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Dosyası", "*.pdf")])
        if path:
            try:
                if ReportGenerator:
                    reporter = ReportGenerator()
                    visuals = {
                        "wordcloud": os.path.abspath("temp_wordcloud.png"),
                        "chart": os.path.abspath("temp_chart.png")
                    }
                    reporter.create_report(path, transcript, active_analyses, report_stats, visuals)
                    messagebox.showinfo("Başarılı", f"Profesyonel Rapor kaydedildi: {os.path.basename(path)}")
                else:
                    # Basit PDF oluşturma (Eğer report_generator modülü yoksa)
                    from fpdf import FPDF
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    tr_map = str.maketrans("ığüşöçİĞÜŞÖÇ", "igusocIGUSOC")
                    clean = text.translate(tr_map).encode('latin-1', 'replace').decode('latin-1')
                    for line in clean.split('\n'):
                        pdf.multi_cell(0, 10, txt=line)
                    pdf.output(path)
                    messagebox.showinfo("Başarılı", "PDF kaydedildi (Sadece Metin).")
            except Exception as e:
                messagebox.showerror("PDF Hatası", f"PDF kaydedilemedi: {e}")

    # --- SİSTEM AYARLARI VE ANAHTAR YÖNETİMİ ---
    def save_api_keys(self):
        """API anahtarlarını .env dosyasına kalıcı ve güvenli olarak kaydeder."""
        openai_key = self.api_entry.get().strip()
        gemini_key = self.gemini_api_entry.get().strip()
        
        try:
            env_path = os.path.join(os.getcwd(), ".env")
            set_key(env_path, "OPENAI_API_KEY", openai_key)
            set_key(env_path, "GEMINI_API_KEY", gemini_key)
        except Exception as e:
            print(f".env kaydetme hatası: {e}")

        # Güvenlik amacıyla eski config.json içindeki anahtarları temizle
        config = {}
        if os.path.exists("config.json"):
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
        
        config.pop("openai_api_key", None)
        config.pop("gemini_api_key", None)
        
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
            
        self.api_key = openai_key
        self.gemini_api_key = gemini_key
        messagebox.showinfo("Başarılı", "API Anahtarları .env dosyasına güvenle kaydedildi.")

    def load_api_key(self):
        """API anahtarlarını önce .env dosyasından, yoksa config.json'dan yükler."""
        try:
            self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
            self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()

            if not self.api_key or not self.gemini_api_key:
                if os.path.exists("config.json"):
                    with open("config.json", "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if not self.api_key:
                            self.api_key = data.get("openai_api_key", "").strip()
                        if not self.gemini_api_key:
                            self.gemini_api_key = data.get("gemini_api_key", "").strip()
            
            # UI giriş alanlarını doldur
            if self.api_key:
                self.api_entry.delete(0, "end")
                self.api_entry.insert(0, self.api_key)
            
            if self.gemini_api_key:
                self.gemini_api_entry.delete(0, "end")
                self.gemini_api_entry.insert(0, self.gemini_api_key)
                        
        except Exception as e:
            print(f"Konfigürasyon yükleme hatası: {e}")

    def on_app_closing(self):
        """Uygulama kapatılırken çalışan temizlik fonksiyonu."""
        self.is_recording = False
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()
