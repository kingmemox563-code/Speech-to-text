"""
audio_recorder.py - Ses Kayıt Modülü
Bu modül, mikrofondan canlı ses alımı, VAD (Ses Aktivite Algılama) kontrolü 
ve ses verilerinin işlenmesi/kaydedilmesi süreçlerini yönetir.
"""

import sounddevice as sd
import numpy as np
import wave
import uuid
import threading
import time
import os
import queue

class AudioRecorder:
    """
    Mikrofon girişini yöneten ve ses verilerini segmentlere ayıran sınıf.
    """
    def __init__(self, transcriber_queue):
        """
        Args:
            transcriber_queue (queue.Queue): İşlenecek ses dosyalarının iletileceği kuyruk.
        """
        self.transcriber_queue = transcriber_queue
        self.is_recording = False
        self.mic_index = None
        self.stream = None
        
        # Whisper için standart değerler
        self.samplerate = 16000
        self.chunk_duration = 5.0  # Her bir ses segmentinin saniye cinsinden süresi
        
        self.buffer = np.zeros((0, 1), dtype=np.float32)
        self.input_queue = queue.Queue()
        self.thread = None
        
        # VAD (Voice Activity Detection - Ses Aktivite Algılama) Ayarları
        # silence_threshold: Sessizlik sınırı. Bu değerin altındaki sesler işlenmez.
        self.silence_threshold = 0.015 
        self.last_chunk = None 
        self.full_recording = [] # Tüm oturumun ham verisi burada tutulur

    def start_recording(self, mic_index):
        """
        Ses kaydını başlatır.
        
        Args:
            mic_index (int): Kullanılacak mikrofonun indeks numarası.
        """
        if self.is_recording:
            return
            
        self.mic_index = mic_index
        self.is_recording = True
        self.full_recording = [] 
        self.buffer = np.zeros((0, 1), dtype=np.float32)
        
        # Ses işleme sürecini ayrı bir thread'de (iş parçacığı) başlat
        self.thread = threading.Thread(target=self._process_audio, daemon=True)
        self.thread.start()

    def stop_recording(self):
        """Ses kaydını durdurur ve kaynakları serbest bırakır."""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.thread:
            self.thread.join(timeout=1.0)

    def save_session_audio(self, filename="session.wav"):
        """
        Tüm oturumu yüksek kaliteli bir WAV dosyası olarak kaydeder.
        Bu dosya daha sonra Gemini gibi modellerle detaylı analiz için kullanılabilir.
        
        Args:
            filename (str): Kaydedilecek dosya adı.
            
        Returns:
            str: Dosyanın mutlak yolu veya hata durumunda None.
        """
        if not self.full_recording:
            return None
            
        try:
            # Tüm parçaları tek bir dizi haline getir
            full_data = np.concatenate(self.full_recording, axis=0)
            
            # Normalizasyon: Ses seviyesini en yüksek noktaya göre ölçeklendir (Ses patlamalarını ve çok düşük sesleri optimize eder)
            max_val = np.max(np.abs(full_data))
            if max_val > 0:
                full_data = full_data / max_val

            # Float veriyi 16-bit tamsayıya çevir (Standart ses dosyası formatı)
            audio_int16 = (full_data * 32767).astype(np.int16)
            
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(1) # Tek kanal (Mono)
                wf.setsampwidth(2) # 16-bit (2 byte)
                wf.setframerate(self.samplerate)
                wf.writeframes(audio_int16.tobytes())
                
            return os.path.abspath(filename)
        except Exception as e:
            print(f"Oturum kaydetme hatası: {e}")
            return None

    def _audio_callback(self, indata, frames, time_info, status):
        """Ham ses verisini mikrofondan alır ve kuyruğa ekler."""
        if status:
            print(f"Ses hatası: {status}")
        self.input_queue.put(indata.copy())

    def _process_audio(self):
        """Arka planda çalışan ana ses işleme döngüsü."""
        try:
            # Mikrofon akışını (stream) başlat
            self.stream = sd.InputStream(samplerate=self.samplerate, 
                                       channels=1, 
                                       callback=self._audio_callback, 
                                       device=self.mic_index)
            with self.stream:
                chunk_samples = int(self.samplerate * self.chunk_duration)
                
                while self.is_recording:
                    try:
                        # Giriş kuyruğundan verileri al ve buffer'a ekle
                        while not self.input_queue.empty():
                            data = self.input_queue.get_nowait()
                            self.full_recording.append(data)
                            self.last_chunk = data
                            self.buffer = np.concatenate((self.buffer, data), axis=0)
                            
                            # Buffer, belirlenen chunk süresine ulaştığında segmenti işle
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
        """Bir ses segmentini kontrol eder (sessizlik ayıklama) ve diske kaydeder."""
        segment_flat = segment.flatten()
        
        # RMS (Root Mean Square) ile sesin enerji seviyesini hesapla (Sessizlik kontrolü)
        rms = np.sqrt(np.mean(segment_flat ** 2))
        if rms < self.silence_threshold:
            # Eğer ses seviyesi eşiğin altındaysa, bu segmenti transkripsiyona gönderme
            return 
            
        # Segment için benzersiz bir geçici dosya adı oluştur
        filename = f"temp_{uuid.uuid4().hex}.wav"
        try:
            # Whisper ve diğer kütüphanelerle uyumluluk için Float32 -> Int16 dönüşümü
            audio_int16 = (segment_flat * 32767).astype(np.int16)
            
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.samplerate)
                wf.writeframes(audio_int16.tobytes())
            
            # Kaydedilen dosyayı Transcriber kuyruğuna (işlenmek üzere) ekle
            self.transcriber_queue.put(filename)
            
        except Exception as e:
            print(f"Segment dosyası yazma hatası: {e}")
