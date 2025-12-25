"""
visualizer.py - Ses Görselleştirme Modülü
Bu modül, mikrofon girişinden gelen ses seviyelerini canlı olarak 
analiz eder ve dinamik barlar (çubuklar) şeklinde görselleştirir.
"""

import customtkinter as ctk
import tkinter as tk
import numpy as np

class AudioVisualizer(ctk.CTkFrame):
    """
    Canlı ses dalgalarını görselleştiren özel bileşen (widget).
    """
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        # Barların çizileceği tuval (canvas)
        self.canvas = tk.Canvas(self, bg="#121212", highlightthickness=0, height=100)
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.bars = 50  # Görselleştiricideki toplam çubuk sayısı
        self.bar_width = 0
        self.rects = [] # Canvas üzerindeki dikdörtgen nesneleri tutar
        self.current_heights = None # Barların o anki yüksekliklerini saklar
        
        # Pencere boyutu değiştiğinde barları yeniden boyutlandır
        self.bind("<Configure>", self._setup_bars)

    def _setup_bars(self, event=None):
        """Canvas üzerindeki barları ilk kez oluşturur veya yeniden boyutlandırır."""
        self.canvas.delete("all")
        self.rects = []
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        if w < 10: w = 600
        
        self.bar_width = w / self.bars
        self.current_heights = np.zeros(self.bars)
        
        for i in range(self.bars):
            # Barın başlangıç (x) konumunu hesapla
            x0 = i * self.bar_width + 1
            x1 = (i + 1) * self.bar_width - 1
            
            # Başlangıçta barlar merkez çizgide (yükseklik 0) durur
            # Modern turkuaz renk paleti kullanılmıştır
            rect = self.canvas.create_rectangle(x0, h/2, x1, h/2, fill="#00adb5", outline="")
            self.rects.append(rect)

    def update_visuals(self, audio_data):
        """
        Gelen ses verisine göre barların yüksekliğini pürüzsüzce günceller.
        
        Args:
            audio_data (np.array): Mikrofondan gelen ham ses verisi.
        """
        if len(self.rects) != self.bars or self.current_heights is None:
            return

        samples = len(audio_data)
        if samples == 0: return
        
        # Ses verisini bar sayısına göre dilimle
        step = samples // self.bars
        h = self.canvas.winfo_height()
        mid_y = h / 2
        
        # --- GÖRSEL PARAMETRELER ---
        decay_rate = 0.12  # Barların düşüş (sönümlenme) hızı
        boost = 45         # Düşük sesleri daha görünür kılan hassasiyet çarpanı
        
        for i in range(self.bars):
            start = i * step
            end = (i + 1) * step
            chunk = audio_data[start:end]
            
            if len(chunk) == 0: continue
            
            # RMS (Enerji) hesapla
            rms = np.sqrt(np.mean(chunk**2))
            
            # Hedef yüksekliği belirle (Canvas boyutuyla orantılı)
            target_h = min(h * 0.95, rms * boost * h) 
            
            # Tepkisellik Kontrolü: Yükselirken anlık, inerken yumuşak geçiş
            if target_h > self.current_heights[i]:
                # Sıçrama (Hızlı yükseliş)
                self.current_heights[i] = target_h
            else:
                # Sönümlenme (Yavaş düşüş)
                self.current_heights[i] -= (self.current_heights[i] - target_h) * decay_rate
            
            val = self.current_heights[i]
            y0 = mid_y - (val / 2) # Üst sınır
            y1 = mid_y + (val / 2) # Alt sınır
            
            # --- DİNAMİK RENKLENDİRME ---
            # Ses şiddetine göre renk değişimi (Yeşil -> Turkuaz -> Beyaz)
            if val < h * 0.3:
                color = "#28a745" # Sakin sesler (Yeşil)
            elif val < h * 0.6:
                color = "#00adb5" # Orta sesler (Turkuaz)
            else:
                color = "#eeeeee" # Yüksek sesler (Parlak beyaz)
            
            x0 = i * self.bar_width + 1
            x1 = (i + 1) * self.bar_width - 1
            
            try:
                # Canvas üzerindeki nesnenin koordinatlarını ve rengini güncelle
                self.canvas.coords(self.rects[i], x0, y0, x1, y1)
                self.canvas.itemconfig(self.rects[i], fill=color)
            except:
                pass

    def clear(self):
        """Barları sıfırlayarak başlangıç konumuna (sessizlik) getirir."""
        h = self.canvas.winfo_height()
        mid_y = h / 2
        if self.current_heights is not None:
            self.current_heights.fill(0)
            for i, r in enumerate(self.rects):
                x0 = i * self.bar_width + 1
                x1 = (i + 1) * self.bar_width - 1
                self.canvas.coords(r, x0, mid_y, x1, mid_y)
