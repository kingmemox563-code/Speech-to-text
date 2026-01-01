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
    def __init__(self, master, mode="neon_bars", **kwargs):
        super().__init__(master, **kwargs)
        self.mode = mode # "bars", "waveform", or "neon_bars"
        # Barların çizileceği tuval (canvas)
        self.canvas = tk.Canvas(self, bg="#121212", highlightthickness=0, height=140)
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.bars = 70  # Daha yoğun neon görünümü
        self.bar_width = 0
        self.rects = [] # Ana barlar
        self.glow_rects = [] # Parlama efekti için arka barlar
        self.line_id = None 
        self.current_heights = None 
        
        self.bind("<Configure>", self._setup_bars)

    def _setup_bars(self, event=None):
        """Canvas üzerindeki barları veya waveform çizgisini hazırlar."""
        self.canvas.delete("all")
        self.rects = []
        self.glow_rects = []
        self.line_id = None
        self.update_idletasks()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        if w < 10: w = self.winfo_width()
        if w < 10: w = 800
        
        self.bar_width = w / self.bars
        self.current_heights = np.zeros(self.bars)
        
        if "bars" in self.mode:
            for i in range(self.bars):
                x0 = i * self.bar_width + 1
                x1 = (i + 1) * self.bar_width - 1
                
                # Parlama efekti (Glow) - Biraz daha geniş ve şeffaf
                glow = self.canvas.create_rectangle(x0-1, h/2, x1+1, h/2, fill="#00adb5", outline="", state="hidden")
                self.glow_rects.append(glow)
                
                # Ana bar
                rect = self.canvas.create_rectangle(x0, h/2, x1, h/2, fill="#00adb5", outline="")
                self.rects.append(rect)
        elif self.mode == "waveform":
            points = []
            for i in range(self.bars):
                points.extend([i * self.bar_width, h/2])
            self.line_id = self.canvas.create_line(points, fill="#00adb5", width=3, smooth=True)

    def update_visuals(self, audio_data):
        """
        Mikrofondan gelen her ses paketinde burası tetikleniyor. 
        Sesi analiz edip ekrandaki barları hareket ettiriyoruz ki kullanıcı sesinin alındığını anlasın.
        """
        if self.current_heights is None: return

        samples = len(audio_data)
        if samples == 0: return
        
        step = samples // self.bars
        h = self.canvas.winfo_height()
        mid_y = h / 2
        
        # --- MODERN ENERJİK PARAMETRELER ---
        decay_rate = 0.18  # Hızlı düşüş (Canlı his)
        boost = 65         # Yüksek hassasiyet
        
        new_points = [] 
        
        for i in range(self.bars):
            start = i * step
            end = (i + 1) * step
            chunk = audio_data[start:end]
            
            if len(chunk) == 0: continue
            
            rms = np.sqrt(np.mean(chunk**2))
            target_h = min(h * 0.9, rms * boost * h) 
            
            if target_h > self.current_heights[i]:
                self.current_heights[i] = target_h
            else:
                self.current_heights[i] -= (self.current_heights[i] - target_h) * decay_rate
            
            val = self.current_heights[i]
            
            # --- DİNAMİK NEON RENKLENDİRME ---
            if val < h * 0.2:
                color, glow_color = "#00adb5", "#007a80" # Cyan
            elif val < h * 0.5:
                color, glow_color = "#bd93f9", "#6272a4" # Purple
            else:
                color, glow_color = "#ff2e63", "#991b3b" # Pink/Red
            
            x0 = i * self.bar_width + 1
            x1 = (i + 1) * self.bar_width - 1
            y0 = mid_y - (val / 2)
            y1 = mid_y + (val / 2)
            
            if "bars" in self.mode and len(self.rects) == self.bars:
                try:
                    # Ana Bar
                    self.canvas.coords(self.rects[i], x0, y0, x1, y1)
                    self.canvas.itemconfig(self.rects[i], fill=color)
                    
                    # Parlama (Opsiyonel: Sadece yüksek seslerde göster)
                    if val > 10:
                        self.canvas.coords(self.glow_rects[i], x0-2, y0-2, x1+2, y1+2)
                        self.canvas.itemconfig(self.glow_rects[i], fill=glow_color, state="normal")
                    else:
                        self.canvas.itemconfig(self.glow_rects[i], state="hidden")
                except: pass
            elif self.mode == "waveform":
                new_points.extend([x0 + self.bar_width/2, y0 if i % 2 == 0 else y1])

        if self.mode == "waveform" and self.line_id:
            try:
                self.canvas.coords(self.line_id, *new_points)
                max_val = np.max(self.current_heights)
                line_color = "#ff2e63" if max_val > h * 0.5 else "#00adb5"
                self.canvas.itemconfig(self.line_id, fill=line_color)
            except: pass

    def clear(self):
        """Barları sıfırlayarak başlangıç konumuna (sessizlik) getirir."""
        h = self.canvas.winfo_height()
        mid_y = h / 2
        if self.current_heights is not None:
            self.current_heights.fill(0)
            for i, r in enumerate(self.rects):
                x0 = i * self.bar_width
                x1 = (i + 1) * self.bar_width
                self.canvas.coords(r, x0, mid_y, x1, mid_y)
