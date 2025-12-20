import customtkinter as ctk
import tkinter as tk
import numpy as np

class AudioVisualizer(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        # Arka planı daha profesyonel, hafif gradyan hissi veren bir renk yapalım
        self.canvas = tk.Canvas(self, bg="#121212", highlightthickness=0, height=100)
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.bars = 50  # Daha yoğun bir görünüm için bar sayısını 50 yaptık
        self.bar_width = 0
        self.rects = []
        self.current_heights = None
        
        self.bind("<Configure>", self._setup_bars)

    def _setup_bars(self, event=None):
        self.canvas.delete("all")
        self.rects = []
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        if w < 10: w = 600 # Varsayılan genişlik iyileştirildi
        
        self.bar_width = w / self.bars
        self.current_heights = np.zeros(self.bars)
        
        for i in range(self.bars):
            x0 = i * self.bar_width + 1
            x1 = (i + 1) * self.bar_width - 1
            
            # Gradyan renk paleti (Turkuazdan koyu maviye)
            # Jüri önünde şık durması için renk geçişi önemli
            rect = self.canvas.create_rectangle(x0, h/2, x1, h/2, fill="#00adb5", outline="")
            self.rects.append(rect)

    def update_visuals(self, audio_data):
        """Ses verisine göre barları ultra hassas şekilde günceller."""
        if len(self.rects) != self.bars or self.current_heights is None:
            return

        samples = len(audio_data)
        if samples == 0: return
        
        step = samples // self.bars
        h = self.canvas.winfo_height()
        mid_y = h / 2
        
        # --- KRİTİK AYARLAR ---
        decay_rate = 0.12  # Çubukların düşüş hızı (Daha akıcı olması için düşürüldü)
        # Hassasiyeti artırdık: RMS değeri genelde 0.01-0.1 arasındadır, bu yüzden çarpanı yükselttik.
        boost = 45         
        
        for i in range(self.bars):
            start = i * step
            end = (i + 1) * step
            chunk = audio_data[start:end]
            
            if len(chunk) == 0: continue
            
            # RMS hesapla ve gürültü tabanını temizle
            rms = np.sqrt(np.mean(chunk**2))
            
            # Logaritmik ölçeklendirme hissi vermek için (Düşük sesleri daha belirgin yapar)
            target_h = min(h * 0.95, rms * boost * h) 
            
            # Tepkisellik (Hızlı yükseliş, yumuşak iniş)
            if target_h > self.current_heights[i]:
                # Yükselirken ani tepki (Hassasiyet hissi)
                self.current_heights[i] = target_h
            else:
                # İnerken sönümleme (Görsel süreklilik)
                self.current_heights[i] -= (self.current_heights[i] - target_h) * decay_rate
            
            val = self.current_heights[i]
            y0 = mid_y - (val / 2)
            y1 = mid_y + (val / 2)
            
            # --- RENK MANTIĞI (Dinamik) ---
            # Ses şiddetine göre renk değişimi (Yeşil -> Turkuaz -> Beyaz)
            if val < h * 0.3:
                color = "#28a745" # Sakin sesler (Yeşil)
            elif val < h * 0.6:
                color = "#00adb5" # Orta sesler (Turkuaz)
            else:
                color = "#eeeeee" # Yüksek sesler (Parlak beyaz/gri)
            
            x0 = i * self.bar_width + 1
            x1 = (i + 1) * self.bar_width - 1
            
            try:
                self.canvas.coords(self.rects[i], x0, y0, x1, y1)
                self.canvas.itemconfig(self.rects[i], fill=color)
            except:
                pass

    def clear(self):
        """Barları pürüzsüzce sıfırlar."""
        h = self.canvas.winfo_height()
        mid_y = h / 2
        if self.current_heights is not None:
            self.current_heights.fill(0)
            for i, r in enumerate(self.rects):
                x0 = i * self.bar_width + 1
                x1 = (i + 1) * self.bar_width - 1
                self.canvas.coords(r, x0, mid_y, x1, mid_y)
