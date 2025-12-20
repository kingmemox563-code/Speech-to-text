import sounddevice as sd
import numpy as np
import wave
import uuid
import threading
import time
import os
import queue

class AudioRecorder:
    def __init__(self, transcriber_queue):
        self.transcriber_queue = transcriber_queue
        self.is_recording = False
        self.mic_index = None
        self.stream = None
        
        # Whisper Standartları
        self.samplerate = 16000
        self.chunk_duration = 5.0  # Doğruluk için uzun tutuldu
        
        self.buffer = np.zeros((0, 1), dtype=np.float32)
        self.input_queue = queue.Queue()
        self.thread = None
        
        # VAD Ayarları
        self.silence_threshold = 0.015 
        self.last_chunk = None 
        self.full_recording = [] 

    def start_recording(self, mic_index):
        if self.is_recording:
            return
            
        self.mic_index = mic_index
        self.is_recording = True
        self.full_recording = [] 
        self.buffer = np.zeros((0, 1), dtype=np.float32)
        
        # Thread başlatma
        self.thread = threading.Thread(target=self._process_audio, daemon=True)
        self.thread.start()

    def stop_recording(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.thread:
            self.thread.join(timeout=1.0)

    def save_session_audio(self, filename="session.wav"):
        """Tüm oturumu Gemini analizi için yüksek kaliteli WAV olarak kaydeder."""
        if not self.full_recording:
            return None
            
        try:
            full_data = np.concatenate(self.full_recording, axis=0)
            # Normalizasyon (Ses seviyesini optimize etme)
            max_val = np.max(np.abs(full_data))
            if max_val > 0:
                full_data = full_data / max_val

            audio_int16 = (full_data * 32767).astype(np.int16)
            
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.samplerate)
                wf.writeframes(audio_int16.tobytes())
                
            return os.path.abspath(filename)
        except Exception as e:
            print(f"Oturum kaydetme hatası: {e}")
            return None

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Ses hatası: {status}")
        self.input_queue.put(indata.copy())

    def _process_audio(self):
        try:
            # InputStream'i with bloğu dışında tanımlayarak daha iyi kontrol ediyoruz
            self.stream = sd.InputStream(samplerate=self.samplerate, 
                                       channels=1, 
                                       callback=self._audio_callback, 
                                       device=self.mic_index)
            with self.stream:
                chunk_samples = int(self.samplerate * self.chunk_duration)
                
                while self.is_recording:
                    try:
                        # Queue'dan veriyi beklemeden al
                        while not self.input_queue.empty():
                            data = self.input_queue.get_nowait()
                            self.full_recording.append(data)
                            self.last_chunk = data
                            self.buffer = np.concatenate((self.buffer, data), axis=0)
                            
                            # Chunk dolduğunda işle
                            if len(self.buffer) >= chunk_samples:
                                segment = self.buffer[:chunk_samples]
                                self.buffer = self.buffer[chunk_samples:]
                                self._handle_segment(segment)
                    except queue.Empty:
                        pass
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"Kayıt akış hatası: {e}")
            self.is_recording = False

    def _handle_segment(self, segment):
        segment_flat = segment.flatten()
        
        # Enerji bazlı sessizlik kontrolü
        rms = np.sqrt(np.mean(segment_flat ** 2))
        if rms < self.silence_threshold:
            return 
            
        # Geçici dosya oluşturma
        filename = f"temp_{uuid.uuid4().hex}.wav"
        try:
            # Float32 -> Int16 dönüşümü (Whisper/Gemini uyumluluğu için)
            audio_int16 = (segment_flat * 32767).astype(np.int16)
            
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.samplerate)
                wf.writeframes(audio_int16.tobytes())
            
            # Transcriber kuyruğuna ekle
            self.transcriber_queue.put(filename)
            
        except Exception as e:
            print(f"Segment dosyası yazma hatası: {e}")
