import whisper
import numpy as np
import os
import queue
import threading
import tempfile
import scipy.io.wavfile as wav
import torch # CUDA kontrolü için gerekli

class Transcriber:
    def __init__(self, device="cpu", model_type="medium"):
        # CUDA tespiti ve model yükleme
        self.device = device
        self.model = whisper.load_model(model_type, device=self.device)
        self.queue = queue.Queue()
        self.is_running = False
        self.audio_buffer = []
        self.current_lang = "turkish"
        self.task = "transcribe"

    def add_audio_chunk(self, chunk):
        self.audio_buffer.extend(chunk.flatten())
        if len(self.audio_buffer) >= 48000:
            self._save_and_queue()

    def _save_and_queue(self):
        data = np.array(self.audio_buffer)
        scaled = (data * 32767).astype(np.int16)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav.write(tmp.name, 16000, scaled)
        self.queue.put(tmp.name)
        self.audio_buffer = []

    def start(self, language="turkish", task="transcribe", callback=None):
        self.is_running = True
        self.current_lang = language
        self.task = task
        self.on_text = callback
        threading.Thread(target=self._worker, daemon=True).start()

    def stop(self):
        self.is_running = False

    def _worker(self):
        while self.is_running:
            try:
                fname = self.queue.get(timeout=1)
                lang_param = None if self.current_lang == "auto" else self.current_lang
                
                # İSTEDİĞİN CUDA VE FP16 DÜZELTMESİ BURADA
                res = self.model.transcribe(
                    fname, 
                    language=lang_param, 
                    task=self.task,
                    fp16=True if self.device == "cuda" else False # GPU varsa True
                )
                
                if res["text"].strip(): 
                    self.on_text(res["text"])
                os.remove(fname)
            except Exception:
                continue
