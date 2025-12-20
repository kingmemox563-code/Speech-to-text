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

# Karakter hatalarını önlemek için sistem dilini UTF-8 yapıyoruz
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    from analytics import AnalyticsGenerator
    from report_generator import ReportGenerator
    from visualizer import AudioVisualizer
except ImportError:
    AnalyticsGenerator = None
    ReportGenerator = None
    AudioVisualizer = None

import datetime

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Donanım ve Durum Ayarları
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_recording = False
        self.audio_frames = []
        self.api_key = ""
        self.fs = 16000 # Whisper standardı
        self.selected_mic_index = self.get_default_mic()
        
        # Son Analiz Verileri
        self.last_analysis = ""
        self.last_transcript = ""
        self.sentiment_stats = {'pos': 33, 'neg': 33, 'neu': 34}
        self.gemini_api_key = ""
        
        # Pencere Konfigürasyonu
        self.title("Ses Analiz Sistemi V16 - Final Pro")
        self.geometry("1300x950")
        ctk.set_appearance_mode("dark")
        self.protocol("WM_DELETE_WINDOW", self.on_app_closing)
        
        self.setup_ui()
        self.load_api_key()
        
    def get_default_mic(self):
        try:
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0:
                    return i
        except:
            return None
        return None

    def get_mic_list(self):
        try:
            devices = sd.query_devices()
            return [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        except:
            return ["No Microphone Found"]

    def setup_ui(self):
        # Klasör oluşturma
        if not os.path.exists("recordings"):
            os.makedirs("recordings")
            
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- SOL PANEL (SideBar) ---
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="SETTINGS", font=("Arial", 20, "bold")).pack(pady=20)
        
        # Donanım Bilgisi
        color = "#00adb5" if self.device == "cuda" else "orange"
        ctk.CTkLabel(self.sidebar, text=f"Hardware: {self.device.upper()}", text_color=color).pack()

        # Whisper Modelleri
        ctk.CTkLabel(self.sidebar, text="Whisper Model:").pack(pady=(10,0))
        self.model_combo = ctk.CTkComboBox(self.sidebar, values=["tiny", "base", "small", "medium"])
        self.model_combo.set("medium")
        self.model_combo.pack(pady=5)

        # Dil Seçimi
        ctk.CTkLabel(self.sidebar, text="Source Language:").pack()
        self.lang_combo = ctk.CTkComboBox(self.sidebar, values=["turkish", "english", "german", "french", "spanish", "italian", "russian"])
        self.lang_combo.set("turkish")
        self.lang_combo.pack(pady=5)

        # Mikrofon Seçimi
        ctk.CTkLabel(self.sidebar, text="Select Microphone:").pack(pady=(10,0))
        self.mic_combo = ctk.CTkComboBox(self.sidebar, values=self.get_mic_list(), command=self.change_mic)
        if self.selected_mic_index is not None:
            mics = self.get_mic_list()
            for m in mics:
                if m.startswith(f"{self.selected_mic_index}:"):
                    self.mic_combo.set(m)
                    break
        self.mic_combo.pack(pady=5)

        # İngilizceye Çeviri Switch
        self.translate_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(self.sidebar, text="Translate to English", variable=self.translate_var).pack(pady=10)

        # Kayıt ve Dosya Butonları
        self.record_btn = ctk.CTkButton(self.sidebar, text="START RECORDING", fg_color="green", command=self.toggle_recording)
        self.record_btn.pack(pady=10, padx=20)

        self.file_btn = ctk.CTkButton(self.sidebar, text="SELECT AUDIO FILE", fg_color="#34495e", command=self.process_audio_file)
        self.file_btn.pack(pady=10, padx=20)

        # Otomatik Kayıt Seçeneği
        self.autosave_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(self.sidebar, text="Auto-save Audio", variable=self.autosave_var).pack(pady=10)

        # Kayıtlar Klasörünü Aç
        self.open_folder_btn = ctk.CTkButton(self.sidebar, text="OPEN RECORDINGS", fg_color="#7f8c8d", command=self.open_recordings_folder)
        self.open_folder_btn.pack(pady=10, padx=20)

        # PDF Kaydetme Özelliği
        self.pdf_btn = ctk.CTkButton(self.sidebar, text="SAVE AS PDF", fg_color="#e67e22", command=self.save_as_pdf)
        self.pdf_btn.pack(pady=10, padx=20)

        # API Girişi
        ctk.CTkLabel(self.sidebar, text="OpenAI API Key:").pack(pady=(10, 0))
        self.api_entry = ctk.CTkEntry(self.sidebar, placeholder_text="OpenAI API Key", show="*")
        self.api_entry.pack(pady=(5, 5), padx=20)
        
        ctk.CTkLabel(self.sidebar, text="Gemini API Key:").pack(pady=(5, 0))
        self.gemini_api_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Gemini API Key", show="*")
        self.gemini_api_entry.pack(pady=(5, 5), padx=20)
        
        ctk.CTkButton(self.sidebar, text="Save Keys", command=self.save_api_keys).pack(pady=5)

        # --- ANA PANEL ---
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=20, pady=10)

        self.status_label = ctk.CTkLabel(self.main_container, text="System Ready", text_color="#00adb5")
        self.status_label.pack(anchor="w")

        # Görselleştirici (Mavi Bar)
        if AudioVisualizer:
            self.visualizer = AudioVisualizer(self.main_container)
            self.visualizer.pack(fill="x", pady=10)

        self.textbox = ctk.CTkTextbox(self.main_container, font=("Consolas", 15))
        self.textbox.pack(fill="both", expand=True, pady=(0, 10))

        # GPT Analiz Butonu
        self.analyze_btn = ctk.CTkButton(self.main_container, text="ANALYZE WITH GPT-4o", fg_color="#10a37f", height=50, command=self.run_analysis)
        self.analyze_btn.pack(fill="x", pady=5)

        # Gemini Analiz Butonu
        self.gemini_analyze_btn = ctk.CTkButton(self.main_container, text="ANALYZE WITH GEMINI", fg_color="#4285f4", height=50, command=self.run_gemini_analysis)
        self.gemini_analyze_btn.pack(fill="x", pady=5)

    def change_mic(self, value):
        try:
            self.selected_mic_index = int(value.split(":")[0])
        except:
            pass

    # --- SES KAYIT VE İŞLEME ---
    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.record_btn.configure(text="STOP RECORDING", fg_color="red")
            self.status_label.configure(text="Recording...", text_color="red")
            self.audio_frames = []
            threading.Thread(target=self._record_thread, daemon=True).start()
        else:
            self.is_recording = False
            self.record_btn.configure(text="START RECORDING", fg_color="green")
            self.status_label.configure(text="Recording stopped.", text_color="#00adb5")
            if hasattr(self, 'visualizer'):
                self.visualizer.clear()

    def _record_thread(self):
        try:
            with sd.InputStream(samplerate=self.fs, channels=1, callback=self._audio_callback, device=self.selected_mic_index):
                while self.is_recording:
                    sd.sleep(100)
        except Exception as e:
            self.is_recording = False
            self.after(0, lambda: messagebox.showerror("Hardware Error", f"Microphone error: {e}"))
            return
        
        
        audio_path = "temp_recording.wav"
        audio_data = np.concatenate(self.audio_frames, axis=0)
        sf.write(audio_path, audio_data, self.fs)

        if self.autosave_var.get():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join("recordings", f"recording_{timestamp}.wav")
            sf.write(save_path, audio_data, self.fs)
            print(f"Audio saved to: {save_path}")

        self._transcribe_file(audio_path)

    def _audio_callback(self, indata, frames, time, status):
        if self.is_recording:
            self.audio_frames.append(indata.copy())
            if hasattr(self, 'visualizer'):
                self.visualizer.update_visuals(indata)

    def _transcribe_file(self, path):
        try:
            from transcriber import Transcriber
            self.status_label.configure(text="Transcribing...")
            
            task = "translate" if self.translate_var.get() else "transcribe"
            model_type = self.model_combo.get()
            
            ts = Transcriber(device=self.device, model_type=model_type)
            res = ts.model.transcribe(path, language=self.lang_combo.get(), task=task)
            
            # Karakter hatasını önlemek için UTF-8 temizliği yapıyoruz
            clean_text = res['text'].encode('utf-8', 'replace').decode('utf-8')
            self.last_transcript = clean_text # PDF için sakla
            # Başlıkta Türkçe karakter kullanmıyoruz (Hatanın asıl çözümü)
            self.after(0, lambda: self.textbox.insert("end", f"\n[TRANSCRIPT]:\n{clean_text}\n"))
            self.status_label.configure(text="Process completed.")
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Transcription Error: {e}"))

    def process_audio_file(self):
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.m4a")])
        if path:
            threading.Thread(target=lambda: self._transcribe_file(path), daemon=True).start()

    def open_recordings_folder(self):
        folder_path = os.path.abspath("recordings")
        if os.path.exists(folder_path):
            os.startfile(folder_path)
        else:
            messagebox.showerror("Error", "Recordings folder not found!")

    # --- GPT-4o ANALİZ (UTF-8 GÜVENLİ) ---
    def run_analysis(self):
        text = self.textbox.get("1.0", "end").strip()
        if text:
            threading.Thread(target=self._gpt_logic, args=(text,), daemon=True).start()

    def _gpt_logic(self, text):
        try:
            if not self.api_key:
                self.after(0, lambda: messagebox.showwarning("Key Missing", "Please save your OpenAI API key."))
                return

            # Karakter temizliği
            safe_text = text.encode('utf-8', 'replace').decode('utf-8')

            client = OpenAI(api_key=self.api_key)
            self.status_label.configure(text="GPT-4o is analyzing...")
            
            prompt = self._get_analysis_prompt(safe_text)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Analyze the transcript(s) professionally and in detail in Turkish. Identify each transcript segment and analyze them both individually and holistically."},
                    {"role": "user", "content": prompt}
                ]
            )
            analysis = response.choices[0].message.content
            self._process_analysis_result(analysis, safe_text, "OpenAI")
        except Exception as e:
            err = str(e).encode('utf-8', 'ignore').decode('utf-8')
            self.after(0, lambda: messagebox.showerror("API Error", f"Error: {err}"))

    # --- GEMINI ANALİZ ---
    def run_gemini_analysis(self):
        text = self.textbox.get("1.0", "end").strip()
        if text:
            threading.Thread(target=self._gemini_logic, args=(text,), daemon=True).start()

    def _gemini_logic(self, text):
        try:
            if not self.gemini_api_key:
                self.after(0, lambda: messagebox.showwarning("Key Missing", "Please save your Gemini API key."))
                return

            # Karakter temizliği
            safe_text = text.encode('utf-8', 'replace').decode('utf-8')

            client = GeminiClient(api_key=self.gemini_api_key)
            self.status_label.configure(text="Gemini is analyzing...")
            
            prompt = self._get_analysis_prompt(safe_text)
            system_instruction = "Analyze the transcript(s) professionally and in detail in Turkish. Identify each transcript segment and analyze them both individually and holistically."
            
            analysis = client.generate_content(prompt, system_instruction=system_instruction)
            self._process_analysis_result(analysis, safe_text, "Gemini")
        except Exception as e:
            err = str(e).encode('utf-8', 'ignore').decode('utf-8')
            self.after(0, lambda: messagebox.showerror("API Error", f"Error: {err}"))

    def _get_analysis_prompt(self, safe_text):
        return f"""
        Sen profesyonel bir veri analisti ve dil bilimcisin. 
        Aşağıdaki metni bir bitirme projesi raporu ciddiyetinde ve derinliğinde analiz et.
        
        BAĞLAM VE GÖREV:
        Bu metin bir veya birden fazla ses kaydının transkriptini içerebilir. Her bir kayıt '[TRANSCRIPT]' başlığı ile belirtilmiştir.
        Senin görevin:
        1. Ekranda kaç farklı veri/kayıt varsa her birini önce KENDİ İÇİNDE analiz et (Tema, Duygu, Önemli Noktalar).
        2. Ardından tüm kayıtları BÜTÜNCÜL olarak ele alıp aralarındaki kontrastı (örneğin biri mutlu diğeri hüzünlü), gelişimleri veya ortak temaları sentezle.
        
        Lütfen şu başlıklar altında ÇOK DETAYLI bir rapor sun:
        
        1. TRANSKRİPT BAZLI ANALİZ: 
           - Kayıt #1: (Konu, baskın duygu ve anahtar ifadeler)
           - Kayıt #2: (Varsa, konu ve karşılaştırmalı duygu)
           - ... (Kaç kayıt varsa devam et)
           
        2. GENEL ÖZET VE SENTEZ: (Tüm kayıtların toplamda ne anlattığı, ruh hali değişimleri ve genel bağlam)
        3. TEMEL KONULAR: (Tüm metinlerde geçen ana konuları madde madde açıkla)
        4. DERİN DUYGU ANALİZİ: (Ses tonu, vurgu, stres düzeyi ve ruh halini hem kayıt bazlı hem genel perspektifle incele. Mutluluk/Hüzün gibi zıtlıkları vurgula)
        5. KRİTİK NOKTALAR VE EYLEM PLANI: (En önemli bulgular ve önerilen adımlar)
        6. ÖNERİLER: (Geliştirme önerileri)
        
        [SKORLAR]:
        (Tüm metnin ağırlıklı duygusunu yansıtan tek bir skor seti)
        POZİTİF: (0-100 arası sayı)
        NEGATİF: (0-100 arası sayı)
        NÖTR: (0-100 arası sayı)
        (Toplam 100 olmalı)
        
        METİN: {safe_text}
        """

    def _process_analysis_result(self, analysis, safe_text, provider):
        # Görsel Analiz (AnalyticsGenerator) Entegrasyonu
        if AnalyticsGenerator:
            try:
                analyzer = AnalyticsGenerator()
                analyzer.generate_wordcloud(safe_text)
                
                pos, neg, neu = 33, 33, 34 
                for line in analysis.split('\n'):
                    if "POZİTİF:" in line: pos = int(''.join(filter(str.isdigit, line)) or 33)
                    if "NEGATİF:" in line: neg = int(''.join(filter(str.isdigit, line)) or 33)
                    if "NÖTR:" in line: neu = int(''.join(filter(str.isdigit, line)) or 34)
                
                self.sentiment_stats = {'pos': pos, 'neg': neg, 'neu': neu}
                analyzer.generate_sentiment_chart(pos, neg, neu)
            except Exception as ae:
                print(f"Visual Analysis Error: {ae}")

        self.last_analysis = analysis 
        self.after(0, lambda: self.textbox.insert("end", f"\n\n[ANALYSIS ({provider})]:\n{analysis}\n"))
        self.status_label.configure(text=f"Analysis finished using {provider}.")

    # --- PDF KAYDETME (PROFESYONEL RAPOR) ---
    def save_as_pdf(self):
        text = self.textbox.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Warning", "Textbox is empty!")
            return
        
        # Eğer henüz analiz yapılmadıysa transcript'i ana metin olarak kullan
        transcript = self.last_transcript if self.last_transcript else text
        analysis = self.last_analysis if self.last_analysis else "Henüz analiz yapılmadı."

        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if path:
            try:
                if ReportGenerator:
                    reporter = ReportGenerator()
                    visuals = {
                        "wordcloud": os.path.abspath("temp_wordcloud.png"),
                        "chart": os.path.abspath("temp_chart.png")
                    }
                    reporter.create_report(path, transcript, analysis, self.sentiment_stats, visuals)
                    messagebox.showinfo("Success", f"Professional Report saved: {os.path.basename(path)}")
                else:
                    # Fallback (Eğer report_generator.py yoksa)
                    from fpdf import FPDF
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    tr_map = str.maketrans("ığüşöçİĞÜŞÖÇ", "igusocIGUSOC")
                    clean = text.translate(tr_map).encode('latin-1', 'replace').decode('latin-1')
                    for line in clean.split('\n'):
                        pdf.multi_cell(0, 10, txt=line)
                    pdf.output(path)
                    messagebox.showinfo("Success", "PDF saved (Text Only).")
            except Exception as e:
                messagebox.showerror("PDF Error", f"Failed to save PDF: {e}")

    # --- SİSTEM AYARLARI ---
    def save_api_keys(self):
        openai_key = self.api_entry.get().strip()
        gemini_key = self.gemini_api_entry.get().strip()
        
        config = {}
        if os.path.exists("config.json"):
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
        
        config["openai_api_key"] = openai_key
        config["gemini_api_key"] = gemini_key
        
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
            
        self.api_key = openai_key
        self.gemini_api_key = gemini_key
        messagebox.showinfo("Success", "API Keys saved.")

    def load_api_key(self):
        try:
            if os.path.exists("config.json"):
                with open("config.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # OpenAI Key
                    raw_openai = data.get("openai_api_key", "").strip()
                    if "sk-proj-" in raw_openai:
                        self.api_key = raw_openai[raw_openai.find("sk-proj-"):].split()[0].replace('"', '').replace('}', '').strip()
                    else:
                        self.api_key = raw_openai
                    
                    if self.api_key:
                        self.api_entry.delete(0, "end")
                        self.api_entry.insert(0, self.api_key)
                        
                    # Gemini Key
                    self.gemini_api_key = data.get("gemini_api_key", "").strip()
                    if self.gemini_api_key:
                        self.gemini_api_entry.delete(0, "end")
                        self.gemini_api_entry.insert(0, self.gemini_api_key)
                        
        except Exception as e:
            print(f"Config load error: {e}")

    def on_app_closing(self):
        self.is_recording = False
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()
