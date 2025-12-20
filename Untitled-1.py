# =================================================================
# AKILLI SES ANALİZ VE DOĞRULAMA SİSTEMİ - ANA UYGULAMA (V4)
# =================================================================
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import sounddevice as sd
import numpy as np
import queue
import whisper
import wave
import time
import noisereduce as nr
import torch
import os
import uuid
import traceback

# Opsiyonel Donanım İzleme
try:
    import psutil
except:
    psutil = None
try:
    import GPUtil
except:
    GPUtil = None

# --- ÖNCEKİ MODÜLLERİNİZDEN IMPORTLAR ---
# Bu dosyaların aynı klasörde olduğunu varsayıyoruz
try:
    from analytics import AnalyticsGenerator
    from report_generator import ReportGenerator
except ImportError:
    print("[!] Uyarı: analytics.py veya report_generator.py bulunamadı. Raporlama devre dışı.")

# ------------------- DONANIM VE MODEL AYARLARI -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
# Doğruluk için 'medium', hız gerekirse 'base' seçilebilir.
MODEL_NAME = "medium" 
print(f"[*] Cihaz: {device.upper()} | Model: {MODEL_NAME} yükleniyor...")
model = whisper.load_model(MODEL_NAME, device=device)

# ------------------- KÜRESEL DEĞİŞKENLER -------------------
SAMPLERATE = 16000
CHUNK_DURATION = 4.0  # Saniye bazlı parça uzunluğu
CHUNK_SAMPLES = int(SAMPLERATE * CHUNK_DURATION)
SILENCE_THRESHOLD = 0.010 # Sessizlik eşiği
is_recording = False
transcription_queue = queue.Queue()
audio_buffer_queue = queue.Queue()
save_folder = "kayitlar"
os.makedirs(save_folder, exist_ok=True)

# ------------------- SES İŞLEME FONKSİYONLARI -------------------
def audio_callback(indata, frames, time_info, status):
    if status: print(f"Ses Hatası: {status}")
    audio_buffer_queue.put(indata.copy())

def audio_stream_thread(mic_index):
    global is_recording
    try:
        with sd.InputStream(samplerate=SAMPLERATE, channels=1, callback=audio_callback, device=mic_index):
            buffer = np.zeros((0, 1), dtype=np.float32)
            while is_recording:
                while not audio_buffer_queue.empty():
                    data = audio_buffer_queue.get().astype(np.float32)
                    buffer = np.concatenate((buffer, data), axis=0)
                    
                    if len(buffer) >= CHUNK_SAMPLES:
                        segment = buffer[:CHUNK_SAMPLES].flatten()
                        buffer = buffer[CHUNK_SAMPLES:]
                        
                        # Sessizlik Algılama (VAD)
                        rms = np.sqrt(np.mean(segment ** 2))
                        if rms < SILENCE_THRESHOLD:
                            continue 
                        
                        # Hafif Gürültü Azaltma
                        reduced = nr.reduce_noise(y=segment, sr=SAMPLERATE, prop_decrease=0.7)
                        
                        # Geçici Dosya Kaydı
                        temp_file = f"temp_{uuid.uuid4().hex}.wav"
                        with wave.open(temp_file, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(SAMPLERATE)
                            wf.writeframes((reduced * 32767).astype(np.int16).tobytes())
                        
                        transcription_queue.put(temp_file)
                time.sleep(0.05)
    except Exception as e:
        print(f"Mikrofon Hatası: {e}")

# ------------------- TRANSKRİPSİYON İŞÇİSİ -------------------
def transcription_worker():
    while is_recording or not transcription_queue.empty():
        try:
            filename = transcription_queue.get(timeout=1)
            
            task = "translate" if translate_var.get() else "transcribe"
            lang = lang_var.get() if lang_var.get() != "auto" else None
            
            result = model.transcribe(
                filename, 
                language=lang, 
                task=task, 
                beam_size=5, 
                temperature=0.0,
                fp16=(device == "cuda")
            )
            
            text = result.get("text", "").strip()
            if text:
                timestamp = time.strftime("[%H:%M:%S] ")
                text_box.after(0, lambda t=f"{timestamp}{text}\n": (text_box.insert(tk.END, t), text_box.see(tk.END)))
            
            if os.path.exists(filename):
                os.remove(filename)
                
        except queue.Empty: continue
        except Exception as e: print(f"Döküm Hatası: {e}")

# ------------------- RAPORLAMA ENTEGRASYONU -------------------
def create_final_report():
    try:
        content = text_box.get("1.0", tk.END).strip()
        if len(content) < 10:
            messagebox.showwarning("Uyarı", "Rapor oluşturmak için yeterli veri yok.")
            return

        # Görselleri oluştur
        analyzer = AnalyticsGenerator()
        wc_path = analyzer.generate_wordcloud(content)
        # Örnek değerler (İleride Gemini'den dinamik alınabilir)
        chart_path = analyzer.generate_sentiment_chart(45, 10, 45) 

        # PDF oluştur
        reporter = ReportGenerator()
        report_name = f"Analiz_Raporu_{int(time.time())}.pdf"
        reporter.create_report(report_name, content, "Otomatik oluşturulmuş konuşma özeti.", 
                              {'pos': 45, 'neg': 10, 'neu': 45}, 
                              {"wordcloud": wc_path, "chart": chart_path})
        
        messagebox.showinfo("Başarılı", f"Rapor Oluşturuldu: {report_name}")
    except Exception as e:
        messagebox.showerror("Hata", f"Rapor oluşturulamadı: {e}")

# ------------------- GUI ARAYÜZÜ -------------------
root = tk.Tk()
root.title("Akıllı Ses Analiz Sistemi - V4")
root.geometry("800x700")
root.configure(bg="#2c3e50")

# Stil Ayarları
style = ttk.Style()
style.theme_use('clam')

# Kontrol Paneli
ctrl_frame = ttk.Frame(root, padding=10)
ctrl_frame.pack(fill=tk.X)

ttk.Label(ctrl_frame, text="Mikrofon:").grid(row=0, column=0, padx=5, sticky="w")
mic_list = [f"{i}: {d['name']}" for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]
mic_combo = ttk.Combobox(ctrl_frame, values=mic_list, width=40, state="readonly")
mic_combo.grid(row=0, column=1, padx=5, pady=5)
if mic_list: mic_combo.current(0)

ttk.Label(ctrl_frame, text="Dil:").grid(row=1, column=0, padx=5, sticky="w")
lang_var = tk.StringVar(value="auto")
lang_combo = ttk.Combobox(ctrl_frame, textvariable=lang_var, values=["auto", "turkish", "english"], state="readonly")
lang_combo.grid(row=1, column=1, padx=5, pady=5, sticky="w")

translate_var = tk.BooleanVar(value=False)
ttk.Checkbutton(ctrl_frame, text="İngilizceye Çevir", variable=translate_var).grid(row=1, column=1, padx=120)

# Butonlar
btn_frame = ttk.Frame(root, padding=10)
btn_frame.pack(fill=tk.X)

def toggle_recording():
    global is_recording
    if not is_recording:
        is_recording = True
        rec_btn.config(text="🔴 Kaydı Durdur", style="Stop.TButton")
        threading.Thread(target=audio_stream_thread, args=(int(mic_combo.get().split(":")[0]),), daemon=True).start()
        threading.Thread(target=transcription_worker, daemon=True).start()
    else:
        is_recording = False
        rec_btn.config(text="🎤 Kaydı Başlat", style="TButton")

rec_btn = ttk.Button(btn_frame, text="🎤 Kaydı Başlat", command=toggle_recording, width=20)
rec_btn.pack(side=tk.LEFT, padx=5)

report_btn = ttk.Button(btn_frame, text="📋 Rapor Oluştur", command=create_final_report)
report_btn.pack(side=tk.LEFT, padx=5)

# Sistem Durumu (CPU/GPU)
sys_label = ttk.Label(root, text="Sistem Hazır", font=("Arial", 9))
sys_label.pack(anchor="e", padx=15)

def update_sys_info():
    if psutil:
        cpu = psutil.cpu_percent()
        gpu_info = ""
        if GPUtil:
            gpus = GPUtil.getGPUs()
            if gpus: gpu_info = f"| GPU: {int(gpus[0].load*100)}%"
        sys_label.config(text=f"CPU: {cpu}% {gpu_info}")
    root.after(2000, update_sys_info)

# Metin Kutusu
text_box = scrolledtext.ScrolledText(root, bg="#1e272e", fg="#ecf0f1", font=("Consolas", 11), padx=10, pady=10)
text_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

update_sys_info()
root.mainloop()