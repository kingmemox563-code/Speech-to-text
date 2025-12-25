"""
transcriber.py - Ses Transkripsiyon (Metne Dönüştürme) Modülü
Bu modül, OpenAI tarafından geliştirilen Whisper modelini kullanarak 
ses dosyalarını yüksek doğrulukla metne dönüştürür.
"""

import whisper
import numpy as np
import os
import queue
import threading
import tempfile
import scipy.io.wavfile as wav
import torch

class Transcriber:
    """
    Ses dosyalarını arka planda metne dönüştüren işleyici sınıf.
    """
    def __init__(self, device="cpu", model_type="medium"):
        """
        Args:
            device (str): "cpu" veya "cuda" (GPU kullanımı için).
            model_type (str): Kullanılacak Whisper model boyutu (tiny, base, small, medium, large).
        """
        self.device = device
        # Modeli hafızaya yükle (Bu işlem model boyutuna göre zaman alabilir)
        self.model = whisper.load_model(model_type, device=self.device)
        self.queue = queue.Queue()
        self.is_running = False
        self.audio_buffer = []
        self.current_lang = "turkish"
        self.task = "transcribe" # "transcribe" (metne dök) veya "translate" (İngilizceye çevir)

    def add_audio_chunk(self, chunk):
        """Ham ses paketlerini buffer'a ekler."""
        self.audio_buffer.extend(chunk.flatten())
        # Buffer yeterli büyüklüğe (yaklaşık 3 saniye) ulaştığında kuyruğa al
        if len(self.audio_buffer) >= 48000:
            self._save_and_queue()

    def _save_and_queue(self):
        """Buffer'daki sesi geçici bir WAV dosyasına kaydeder ve işleme kuyruğuna ekler."""
        data = np.array(self.audio_buffer)
        # Float veriyi 16-bit PCM formatına dönüştür
        scaled = (data * 32767).astype(np.int16)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav.write(tmp.name, 16000, scaled)
        self.queue.put(tmp.name)
        self.audio_buffer = []

    def start(self, language="turkish", task="transcribe", callback=None):
        """Transkripsiyon işçisini (worker) başlatır."""
        self.is_running = True
        self.current_lang = language
        self.task = task
        self.on_text = callback
        # Arka planda çalışacak thread'i başlat
        threading.Thread(target=self._worker, daemon=True).start()

    def stop(self):
        """İşleyiciyi durdurur."""
        self.is_running = False

    def _worker(self):
        """Kuyruktaki ses dosyalarını sırayla işleyen döngü."""
        while self.is_running:
            try:
                # Kuyruktan dosya yolunu al (1 saniye bekle)
                fname = self.queue.get(timeout=1)
                lang_param = None if self.current_lang == "auto" else self.current_lang
                
                # Whisper modelini kullanarak sesi metne dönüştür
                res = self.model.transcribe(
                    fname, 
                    language=lang_param, 
                    task=self.task,
                    # GPU varsa FP16 (hızlı mod) kullan
                    fp16=True if self.device == "cuda" else False 
                )
                
                # Eğer metin boş değilse callback fonksiyonunu çağır (UI'ya yazı gönderir)
                if res["text"].strip(): 
                    self.on_text(res["text"])
                
                # İşlem bitince geçici dosyayı sil
                os.remove(fname)
            except Exception:
                # Kuyruk boşsa veya hata oluşursa döngüye devam et
                continue
