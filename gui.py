"""
gui.py - Ana Kullanıcı Arayüzü (GUI) Modülü
Bu modül, uygulamanın görsel arayüzünü (CustomTkinter), ses kayıt kontrollerini, 
API entegrasyonlarını ve raporlama özelliklerini bir araya getirir.
"""

import customtkinter as ctk
from tkinter import messagebox, filedialog, ttk
import threading
import queue
import sounddevice as sd
import soundfile as sf
import json
import torch
import os
import numpy as np
from openai import OpenAI
from fpdf import FPDF
from docx import Document
from docx.shared import Inches
from gemini_client import GeminiClient
from dotenv import load_dotenv, set_key
import datetime
import whisper
import noisereduce as nr
import pygame
import os
import shutil
import pywinstyles # Modern Windows pencere efektleri için

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

class SentimentTimeline(ctk.CTkFrame):
    """Analiz sekmesi için etkileşimli duygu zaman çizelgesi."""
    def __init__(self, master, textbox_to_scroll, **kwargs):
        super().__init__(master, **kwargs)
        self.textbox = textbox_to_scroll
        self.segments = []
        self.canvas = ctk.CTkCanvas(self, height=40, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(fill="x", padx=10, pady=5)
        # Tıklama olayı geri yüklendi (User request)
        self.canvas.bind("<Button-1>", self._on_click)
        self.tooltip = None

    def update_timeline(self, segments):
        """
        segments: list of dicts like [{"text": "...", "sentiment": "pos/neg/neu", "index": float}]
        """
        self.segments = segments
        self.canvas.delete("all")
        if not segments: return

        width = self.canvas.winfo_width()
        if width <= 1: width = 600 # Fallback width

        total_length = sum(len(s["text"]) for s in segments)
        current_x = 0
        
        colors = {"pos": "#2ecc71", "neg": "#e74c3c", "neu": "#95a5a6"}
        
        for i, seg in enumerate(segments):
            seg_len = len(seg["text"])
            seg_width = (seg_len / total_length) * width
            
            x1 = current_x
            x2 = current_x + seg_width
            
            color = colors.get(seg.get("sentiment", "neu"), "#95a5a6")
            self.canvas.create_rectangle(x1, 5, x2, 35, fill=color, outline="", tags=f"seg_{i}")
            
            current_x += seg_width

    def _on_click(self, event):
        if not self.segments: return
        
        width = self.canvas.winfo_width()
        click_ratio = event.x / width
        
        total_text = "".join(s["text"] for s in self.segments)
        target_char_idx = int(click_ratio * len(total_text))
        
        # Metin kutusunda ilgili bölgeye ilerle
        current_char_count = 0
        for seg in self.segments:
            current_char_count += len(seg["text"])
            if current_char_count >= target_char_idx:
                # Metni bul ve yanıp sönme efektini yap (Opsiyonel)
                search_text = seg["text"][:30] # İlk 30 karakteri ara
                idx = self.textbox.search(search_text, "1.0", "end")
                if idx:
                    self.textbox.see(idx)
                    self.textbox.tag_add("highlight", idx, f"{idx} + {len(search_text)} chars")
                    self.textbox.tag_config("highlight", background="#00adb5", foreground="white")
                    self.after(1000, lambda: self.textbox.tag_remove("highlight", "1.0", "end"))
                break

class MicroAnimation:
    """Durum çubuğu için küçük, şık animasyonlar."""
    def __init__(self, parent_label):
        self.label = parent_label
        self.original_text = parent_label.cget("text")
        self.anim_running = False
        self.dots = 0

    def start_loading(self, text=None):
        if text: self.original_text = text
        self.anim_running = True
        self._animate_dots()

    def start_pulse(self):
        self.anim_running = True
        self._animate_pulse(0)

    def stop(self, final_text="Sistem Hazır"):
        self.anim_running = False
        self.label.configure(text=final_text)

    def _animate_dots(self):
        if not self.anim_running: return
        self.dots = (self.dots + 1) % 4
        self.label.configure(text=f"{self.original_text}{'.' * self.dots}")
        self.label.after(500, self._animate_dots)

    def _animate_pulse(self, step):
        if not self.anim_running: return
        # CustomTkinter'da direct alpha yok, alternatif renk değişimi
        colors = ["#00adb5", "#008a91", "#00676d", "#004449", "#00676d", "#008a91"]
        self.label.configure(text_color=colors[step % len(colors)])
        self.label.after(150, lambda: self._animate_pulse(step + 1))

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
        self.audio_queue = queue.Queue() # Ses verileri için iş parçacığı güvenli kuyruk
        self.all_session_transcripts = [] # Oturum boyuncaki tüm transkriptleri saklayan liste
        
        # Pygame Mixer Başlat (TTS ve Çalma için)
        try:
            pygame.mixer.init()
        except:
            print("Pygame mixer başlatılamadı.")
        
        # Son Analiz ve Transkript Verileri
        self.last_analysis = ""
        self.last_transcript = ""
        self.sentiment_stats = {'pos': 33, 'neg': 33, 'neu': 34}
        self.gemini_api_key = ""
        
        # Whisper Model Önbelleği
        self.whisper_model = None
        self.current_model_type = None
        
        # Ana Pencere Konfigürasyonu
        self.title("Ses Analiz Sistemi")
        self.geometry("1300x950")
        ctk.set_appearance_mode("dark") # Koyu tema varsayılan
        self.protocol("WM_DELETE_WINDOW", self.on_app_closing)
        
        # Arayüzü oluştur ve kayıtlı anahtarları yükle
        self.setup_ui()
        self.load_api_key()

        # Animasyon Yöneticisi
        self.animator = MicroAnimation(self.status_label)
        
        # Windows Modern Efektlerini Uygula (Glassmorphism)
        try:
            # Arka planı koyu ve pürüzsüz yap
            pywinstyles.apply_style(self, "mica")
            # Sol menüye hafif bir opaklık ver
            pywinstyles.set_opacity(self.navigation_frame, value=0.9)
        except Exception as pe:
            print(f"Pencere stili hatası: {pe}")
        
        # Analiz Sonuçlarını Saklama (Çoklu PDF raporu için)
        self.analysis_results = {"OpenAI": "", "Gemini": ""}
        self.all_sentiment_stats = {"OpenAI": None, "Gemini": None}
        
        # Dil Öğrenme (Language Coach) Durumu
        self.target_language = "İngilizce"
        self.user_level = "A2 (Gelişmekte Olan)"
        self.coach_mode = "Serbest Konuşma"
        self.language_analysis_result = ""
        
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
        """Tüm görsel bileşenleri (butonlar, sekmeler, paneller) modernize edilmiş şekilde oluşturur."""
        # Kayıtlar klasörü yoksa oluştur
        if not os.path.exists("recordings"):
            os.makedirs("recordings")
            
        # Grid (Izgara) sistemini yapılandır (Sidebar ve Main Container)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- YAN NAVİGASYON PANELİ (Navigation Sidebar) ---
        self.navigation_frame = ctk.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(5, weight=1)

        self.navigation_frame_label = ctk.CTkLabel(self.navigation_frame, text=" SES ANALİZ\nSİSTEMİ", 
                                                 font=ctk.CTkFont(size=20, weight="bold"),
                                                 text_color="#00adb5") # Turkuaz vurgu
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        # Navigasyon Butonları
        self.home_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Dashboard",
                                        fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                        anchor="w", command=self.home_button_event)
        self.home_button.grid(row=1, column=0, sticky="ew")

        self.analysis_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Analiz Raporu",
                                            fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                            anchor="w", command=self.analysis_button_event)
        self.analysis_button.grid(row=2, column=0, sticky="ew")

        self.history_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Geçmiş Kayıtlar",
                                          fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                          anchor="w", command=self.history_button_event)
        self.history_button.grid(row=3, column=0, sticky="ew")

        self.language_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Dil Koçu (AI)",
                                            fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                            anchor="w", command=self.language_button_event)
        self.language_button.grid(row=4, column=0, sticky="ew")

        self.settings_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Ayarlar",
                                            fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                            anchor="w", command=self.settings_button_event)
        self.settings_button.grid(row=5, column=0, sticky="ew")

        # Görünüm Menüsü (Sidebar Alt Kısmı)
        self.appearance_mode_menu = ctk.CTkOptionMenu(self.navigation_frame, values=["Dark", "Light", "System"],
                                                    command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=6, column=0, padx=20, pady=20, sticky="s")

        # --- ANA İÇERİK PANELLERİ ---
        
        # 1. DASHBOARD PANELİ (Ana Kayıt Ekranı)
        self.home_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.home_frame.grid_columnconfigure(0, weight=1)
        self.home_frame.grid_rowconfigure(2, weight=1)

        # 1. Ses Görselleştirici (En Üst)
        self.viz_container = ctk.CTkFrame(self.home_frame, corner_radius=15)
        self.viz_container.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        if AudioVisualizer:
            self.visualizer = AudioVisualizer(self.viz_container, mode="neon_bars") # Modern neon barlar
            self.visualizer.pack(fill="x", padx=2, pady=5)
        else:
            ctk.CTkLabel(self.viz_container, text="Görselleştirici Modülü Yüklenemedi").pack(pady=20)

        # 2. Durum Çubuğu (Barın Altında)
        self.status_bar = ctk.CTkFrame(self.home_frame, height=40, corner_radius=10)
        self.status_bar.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="ew")
        
        self.status_label = ctk.CTkLabel(self.status_bar, text="Sistem Hazır", text_color="#00adb5", font=("Arial", 13, "bold"))
        self.status_label.pack(side="left", padx=20)

        color = "#00adb5" if self.device == "cuda" else "orange"
        ctk.CTkLabel(self.status_bar, text=f"Donanım: {self.device.upper()}", text_color=color).pack(side="right", padx=20)

        # Transkript Alanı
        self.textbox = ctk.CTkTextbox(self.home_frame, font=("Consolas", 15), corner_radius=15, border_width=2)
        self.textbox.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")

        # Kontrol Butonları (Dashboard)
        self.dashboard_controls = ctk.CTkFrame(self.home_frame, fg_color="transparent")
        self.dashboard_controls.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.dashboard_controls.grid_columnconfigure((0, 1), weight=1)

        self.record_btn = ctk.CTkButton(self.dashboard_controls, text="KAYDI BAŞLAT", fg_color="green", font=("Arial", 14, "bold"),
                                       height=50, command=self.toggle_recording)
        self.record_btn.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        self.file_btn = ctk.CTkButton(self.dashboard_controls, text="SES DOSYASI YÜKLE", fg_color="#34495e", font=("Arial", 14, "bold"),
                                     height=50, command=self.process_audio_file)
        self.file_btn.grid(row=0, column=1, padx=(5, 0), sticky="ew")

        # 2. ANALİZ PANELİ (Detaylı AI Geri Bildirimi)
        self.analysis_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.analysis_frame.grid_columnconfigure(0, weight=3) # Metin alanı
        self.analysis_frame.grid_columnconfigure(1, weight=2) # Görsel alanı
        self.analysis_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(self.analysis_frame, text="YAPAY ZEKA ANALİZ SONUÇLARI", font=("Arial", 22, "bold")).grid(row=0, column=0, columnspan=2, pady=20)

        # Analiz Metin Kutusu
        self.analysis_textbox = ctk.CTkTextbox(self.analysis_frame, font=("Consolas", 14), corner_radius=15)
        self.analysis_textbox.grid(row=1, column=0, padx=(20, 10), pady=10, sticky="nsew")

        # Görsel Alanı (Pie Chart & WordCloud)
        self.viz_frame = ctk.CTkScrollableFrame(self.analysis_frame, corner_radius=15, label_text="Görsel Bilgi Kartları")
        self.viz_frame.grid(row=1, column=1, padx=(10, 20), pady=10, sticky="nsew")

        self.sentiment_img_label = ctk.CTkLabel(self.viz_frame, text="Duygu Analizi Henüz Yapılmadı")
        self.sentiment_img_label.pack(pady=10)

        # Sentiment Timeline (Yeni)
        self.timeline_label = ctk.CTkLabel(self.viz_frame, text="Zaman Bazlı Duygu Dağılımı (Tıklanabilir):", font=("Arial", 11, "bold"))
        self.timeline_label.pack(pady=(10, 0))
        self.sentiment_timeline = SentimentTimeline(self.viz_frame, self.analysis_textbox, height=50, fg_color="transparent")
        self.sentiment_timeline.pack(fill="x", padx=5)

        self.wordcloud_img_label = ctk.CTkLabel(self.viz_frame, text="Kelime Bulutu Henüz Oluşturulmadı")
        self.wordcloud_img_label.pack(pady=10)

        self.chat_display = ctk.CTkTextbox(self.analysis_frame, height=250)
        self.chat_display.grid(row=3, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")

        # Hızlı Aksiyon Butonları
        self.btn_row = ctk.CTkFrame(self.analysis_frame, fg_color="transparent")
        self.btn_row.grid(row=4, column=0, columnspan=2, padx=20, pady=(0, 10), sticky="ew")
        self.btn_row.grid_columnconfigure((0, 1, 2), weight=1)
        
        self.btn_summary = ctk.CTkButton(self.btn_row, text="📋 Özetle", width=100, command=lambda: self._send_quick_chat("Bu konuşmanın kısa ve etkili bir özetini çıkar."))
        self.btn_summary.grid(row=0, column=0, padx=5, sticky="ew")
        
        self.btn_points = ctk.CTkButton(self.btn_row, text="🎯 Kritik Noktalar", width=120, command=lambda: self._send_quick_chat("Bu konuşmadaki en önemli 3 kritik noktayı maddeler halinde yaz."))
        self.btn_points.grid(row=0, column=1, padx=5, sticky="ew")
        
        self.btn_tts = ctk.CTkButton(self.btn_row, text="🔊 Yanıtı Seslendir", width=130, command=self._speak_last_response, fg_color="#ff5722", hover_color="#e64a19")
        self.btn_tts.grid(row=0, column=2, padx=5, sticky="ew")

        # Analiz Başlatma Butonları (Analiz Sekmesi Üstü)
        self.analysis_actions = ctk.CTkFrame(self.analysis_frame, fg_color="transparent")
        self.analysis_actions.grid(row=2, column=0, columnspan=2, padx=20, pady=(10, 0), sticky="ew")
        self.analysis_actions.grid_columnconfigure((0, 1, 2), weight=1)

        self.analyze_btn = ctk.CTkButton(self.analysis_actions, text="GPT-4o İLE ANALİZ ET", fg_color="#10a37f", height=45, command=self.run_analysis)
        self.analyze_btn.grid(row=0, column=0, padx=5, sticky="ew")

        self.gemini_analyze_btn = ctk.CTkButton(self.analysis_actions, text="GEMINI İLE ANALİZ ET", fg_color="#4285f4", height=45, command=self.run_gemini_analysis)
        self.gemini_analyze_btn.grid(row=0, column=1, padx=5, sticky="ew")

        self.export_btn = ctk.CTkButton(self.analysis_actions, text="RAPORU DIŞA AKTAR", fg_color="#e67e22", height=45, command=self.export_results)
        self.export_btn.grid(row=0, column=2, padx=5, sticky="ew")

        # --- AI CHAT (SORU-CEVAP) BÖLÜMÜ ---
        self.chat_frame = ctk.CTkFrame(self.analysis_frame, corner_radius=15, border_width=1, border_color="#00adb5")
        self.chat_frame.grid(row=5, column=0, columnspan=2, padx=20, pady=(0, 20), sticky="ew")
        
        ctk.CTkLabel(self.chat_frame, text="AI'ya Sor:", font=("Arial", 12, "bold")).pack(side="left", padx=10, pady=10)
        self.chat_entry = ctk.CTkEntry(self.chat_frame, placeholder_text="Bu konuşmadan ne öğrenmek istersin?", height=35)
        self.chat_entry.pack(side="left", fill="x", expand=True, padx=10, pady=10)
        
        self.ask_btn = ctk.CTkButton(self.chat_frame, text="SOR", width=80, height=35, fg_color="#ff2e63", command=self.ask_ai_question)
        self.ask_btn.pack(side="right", padx=10, pady=10)
        
        # Soru kutusunda Enter tuşuna basınca soruyu gönder
        self.chat_entry.bind("<Return>", lambda e: self.ask_ai_question())

        # ttk.Treeview stilini güncelle (CustomTkinter ile uyum için)
        from tkinter import ttk
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#2b2b2b", foreground="white", fieldbackground="#2b2b2b", borderwidth=0)
        style.map("Treeview", background=[('selected', '#00adb5')])

        # 3. GEÇMİŞ PANELİ
        self.history_frame = ctk.CTkFrame(self, fg_color="transparent")
        ctk.CTkLabel(self.history_frame, text="KAYIT GEÇMİŞİ", font=("Arial", 22, "bold")).pack(pady=20)
        
        # Geçmiş Tablosu
        self.history_table = ttk.Treeview(self.history_frame, columns=("Tarih", "Model", "Özet", "İşlem"), show="headings")
        self.history_table.heading("Tarih", text="Tarih")
        self.history_table.heading("Model", text="Model")
        self.history_table.heading("Özet", text="Özet")
        self.history_table.heading("İşlem", text="İşlem")
        self.history_table.pack(fill="both", expand=True, padx=20, pady=20)
        self.history_table.bind("<Double-1>", self._on_history_click) # Çift tıklama ile oynat
        self.history_table.bind("<<TreeviewSelect>>", self._on_history_click) # Seçimle de tetiklenebilir

        self.refresh_history_btn = ctk.CTkButton(self.history_frame, text="GEÇMİŞİ YENİLE", command=self.update_history_list)
        self.refresh_history_btn.pack(pady=20)

        # 4. DİL KOÇU (AI LANGUAGE COACH) PANELİ
        self.language_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.language_frame.grid_columnconfigure(0, weight=1)
        self.language_frame.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(self.language_frame, text="AI DİL KOÇU & MENTOR", font=("Arial", 22, "bold"), text_color="#00adb5").grid(row=0, column=0, pady=(20, 10))

        # Dil Ayarları Üst Bar
        self.lang_coach_settings = ctk.CTkFrame(self.language_frame)
        self.lang_coach_settings.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(self.lang_coach_settings, text="Hedef Dil:").pack(side="left", padx=10, pady=10)
        self.coach_lang_combo = ctk.CTkComboBox(self.lang_coach_settings, values=["İngilizce", "Almanca", "Fransızca", "İspanyolca", "İtalyanca", "Rusça"])
        self.coach_lang_combo.set("İngilizce")
        self.coach_lang_combo.pack(side="left", padx=5)

        ctk.CTkLabel(self.lang_coach_settings, text="Seviye:").pack(side="left", padx=10)
        self.coach_level_combo = ctk.CTkComboBox(self.lang_coach_settings, values=["A1 (Başlangıç)", "A2 (Temel)", "B1 (Orta)", "B2 (Üst Orta)", "C1 (İleri)"])
        self.coach_level_combo.set("A2 (Temel)")
        self.coach_level_combo.pack(side="left", padx=5)

        ctk.CTkLabel(self.lang_coach_settings, text="Mod:").pack(side="left", padx=10)
        self.coach_mode_combo = ctk.CTkComboBox(self.lang_coach_settings, values=["Serbest Konuşma", "Gramatik Düzeltme", "Kelime Dağarcığı Geliştirme"])
        self.coach_mode_combo.set("Serbest Konuşma")
        self.coach_mode_combo.pack(side="left", padx=5)

        # Dil Koçu Geri Bildirim Alanı
        self.language_textbox = ctk.CTkTextbox(self.language_frame, font=("Consolas", 15), corner_radius=15, border_width=2, border_color="#00adb5")
        self.language_textbox.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        
        # başlangıç mesajı
        self.language_textbox.insert("1.0", "--- AI DİL KOÇU HAZIR ---\nLütfen bir ses kaydı yapın veya metin girin, ardından 'DİL ANALİZİ BAŞLAT' butonuna basın.\n")

        # Aksiyon Butonları
        self.coach_actions = ctk.CTkFrame(self.language_frame, fg_color="transparent")
        self.coach_actions.grid(row=3, column=0, padx=20, pady=20, sticky="ew")
        self.coach_actions.grid_columnconfigure((0, 1), weight=1)

        self.run_coach_btn = ctk.CTkButton(self.coach_actions, text="🚀 DİL ANALİZİ BAŞLAT", fg_color="#00adb5", font=("Arial", 14, "bold"),
                                          height=50, command=self.run_language_analysis)
        self.run_coach_btn.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        self.speak_coach_btn = ctk.CTkButton(self.coach_actions, text="🔊 DÜZELTMELERİ SESLENDİR", fg_color="#ff5722", font=("Arial", 14, "bold"),
                                            height=50, command=self._speak_language_response)
        self.speak_coach_btn.grid(row=0, column=1, padx=(5, 0), sticky="ew")

        # 5. AYARLAR PANELİ
        self.settings_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.settings_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(self.settings_frame, text="SİSTEM AYARLARI", font=("Arial", 22, "bold")).grid(row=0, column=0, pady=20)

        # API Ayarları Grubu
        self.api_group = ctk.CTkFrame(self.settings_frame)
        self.api_group.grid(row=1, column=0, padx=40, pady=10, sticky="ew")
        
        ctk.CTkLabel(self.api_group, text="API YAPILANDIRMASI", font=("Arial", 14, "bold")).pack(pady=10)
        
        ctk.CTkLabel(self.api_group, text="OpenAI API Anahtarı:").pack()
        self.api_entry = ctk.CTkEntry(self.api_group, width=400, show="*")
        self.api_entry.pack(pady=5)

        ctk.CTkLabel(self.api_group, text="Gemini API Anahtarı:").pack()
        self.gemini_api_entry = ctk.CTkEntry(self.api_group, width=400, show="*")
        self.gemini_api_entry.pack(pady=5)

        ctk.CTkButton(self.api_group, text="Anahtarları Güvenli Kaydet", command=self.save_api_keys).pack(pady=15)

        # Model Ayarları Grubu
        self.model_group = ctk.CTkFrame(self.settings_frame)
        self.model_group.grid(row=2, column=0, padx=40, pady=10, sticky="ew")
        
        ctk.CTkLabel(self.model_group, text="MODEL VE DONANIM", font=("Arial", 14, "bold")).pack(pady=10)

        # Grid for model settings
        model_grid = ctk.CTkFrame(self.model_group, fg_color="transparent")
        model_grid.pack(pady=5)

        ctk.CTkLabel(model_grid, text="Whisper Modeli:").grid(row=0, column=0, padx=10)
        self.model_combo = ctk.CTkComboBox(model_grid, values=["tiny", "base", "small", "medium", "large-v3"])
        self.model_combo.set("medium")
        self.model_combo.grid(row=0, column=1, pady=5)

        ctk.CTkLabel(model_grid, text="Kaynak Dil:").grid(row=1, column=0, padx=10)
        self.lang_options = {
            "Otomatik Algıla": None,
            "Türkçe": "turkish", 
            "İngilizce": "english", 
            "Almanca": "german", 
            "Fransızca": "french", 
            "İspanyolca": "spanish", 
            "İtalyanca": "italian", 
            "Rusça": "russian"
        }
        self.lang_combo = ctk.CTkComboBox(model_grid, values=list(self.lang_options.keys()))
        self.lang_combo.set("Türkçe")
        self.lang_combo.grid(row=1, column=1, pady=5)

        ctk.CTkLabel(model_grid, text="Mikrofon:").grid(row=2, column=0, padx=10)
        self.mic_combo = ctk.CTkComboBox(model_grid, values=self.get_mic_list(), command=self.change_mic)
        self.mic_combo.grid(row=2, column=1, pady=5)

        ctk.CTkLabel(model_grid, text="Yapay Zeka Sesi:").grid(row=3, column=0, padx=10)
        self.tts_voices = {
            "Profesyonel Erkek (Onyx)": "onyx",
            "Sert Erkek (Echo)": "echo",
            "Genç ve Nazik (Nova)": "nova",
            "Net ve Parlak (Shimmer)": "shimmer",
            "Dışavurumcu (Fable)": "fable",
            "Dengeli ve Nötr (Alloy)": "alloy"
        }
        self.tts_voice_combo = ctk.CTkComboBox(model_grid, values=list(self.tts_voices.keys()))
        self.tts_voice_combo.set("Genç ve Nazik (Nova)")
        self.tts_voice_combo.grid(row=3, column=1, pady=5)

        ctk.CTkLabel(model_grid, text="AI Karakteri:").grid(row=4, column=0, padx=10)
        self.personas = {
            "Profesyonel Analist": "analyst",
            "Utangaç ve Cıvıl Cıvıl": "shy_girl"
        }
        self.persona_combo = ctk.CTkComboBox(model_grid, values=list(self.personas.keys()))
        self.persona_combo.set("Profesyonel Analist")
        self.persona_combo.grid(row=4, column=1, pady=5)

        # Switchler
        self.translate_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(self.model_group, text="Tanımadan Sonra İngilizceye Çevir", variable=self.translate_var).pack(pady=5)
        
        self.autosave_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(self.model_group, text="Ses Kayıtlarını Otomatik Arşivle", variable=self.autosave_var).pack(pady=5)

        self.noise_reduce_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(self.model_group, text="Gelişmiş Gürültü Azaltma (Önerilen)", variable=self.noise_reduce_var).pack(pady=5)

        # Varsayılan Sayfayı Göster
        self.select_frame_by_name("home")

    def select_frame_by_name(self, name):
        # Buton renklerini sıfırla
        self.home_button.configure(fg_color=("gray75", "gray25") if name == "home" else "transparent")
        self.analysis_button.configure(fg_color=("gray75", "gray25") if name == "analysis" else "transparent")
        self.history_button.configure(fg_color=("gray75", "gray25") if name == "history" else "transparent")
        self.language_button.configure(fg_color=("gray75", "gray25") if name == "language" else "transparent")
        self.settings_button.configure(fg_color=("gray75", "gray25") if name == "settings" else "transparent")

        # Sayfaları gizle
        self.home_frame.grid_forget()
        self.analysis_frame.grid_forget()
        self.history_frame.grid_forget()
        self.language_frame.grid_forget()
        self.settings_frame.grid_forget()

        # Seçilen sayfayı göster
        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        elif name == "analysis":
            self.analysis_frame.grid(row=0, column=1, sticky="nsew")
        elif name == "history":
            self.history_frame.grid(row=0, column=1, sticky="nsew")
            self.update_history_list()
        elif name == "language":
            self.language_frame.grid(row=0, column=1, sticky="nsew")
        elif name == "settings":
            self.settings_frame.grid(row=0, column=1, sticky="nsew")

    def home_button_event(self):
        self.select_frame_by_name("home")

    def analysis_button_event(self):
        self.select_frame_by_name("analysis")

    def history_button_event(self):
        self.select_frame_by_name("history")

    def language_button_event(self):
        self.select_frame_by_name("language")

    def settings_button_event(self):
        self.select_frame_by_name("settings")

    def change_appearance_mode_event(self, new_appearance_mode):
        ctk.set_appearance_mode(new_appearance_mode)

    def update_history_list(self):
        """Kayıtlar klasöründeki dosyaları listeler ve tabloyu günceller."""
        # Tabloyu temizle
        for item in self.history_table.get_children():
            self.history_table.delete(item)
            
        if not os.path.exists("recordings"):
            os.makedirs("recordings")
            
        recordings = sorted(os.listdir("recordings"), reverse=True)
        recordings = [f for f in recordings if f.endswith(".wav")]
        
        for filename in recordings:
            # Tarih bilgisini dosyadan çıkar (Format: kayit_20231227_120000.wav)
            date_info = filename.replace("kayit_", "").replace(".wav", "").replace("_", " ")
            self.history_table.insert("", "end", values=(
                date_info, 
                "Whisper", 
                f"{filename}", 
                "OYNAT ▶️"
            ))

    def _on_history_click(self, event):
        """Geçmiş tablosuna tıklandığında kaydı oynatır."""
        selected = self.history_table.selection()
        if not selected: return
        
        item_values = self.history_table.item(selected[0])["values"]
        filename = item_values[2] # Özet/Filename kolonu
        path = os.path.join("recordings", filename)
        
        if os.path.exists(path):
            self._play_audio(path)
        else:
            messagebox.showinfo("Bilgi", "Ses dosyası bulunamadı.")

    def _play_audio(self, file_path):
        """Verilen ses dosyasını pygame ile çalar."""
        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
        except Exception as e:
            messagebox.showerror("Hata", f"Ses oynatılamadı: {e}")

    def _speak_last_response(self):
        """Son AI yanıtını OpenAI TTS kullanarak seslendirir (Hyper-realistic)."""
        if not self.last_analysis:
            messagebox.showwarning("Uyarı", "Seslendirilecek bir yanıt yok.")
            return
            
        threading.Thread(target=self._tts_worker, daemon=True).start()

    def _tts_worker(self):
        """TTS işlemini arka planda yapar."""
        try:
            if not self.api_key:
                self.after(0, lambda: messagebox.showerror("Hata", "OpenAI API anahtarı bulunamadı."))
                return

            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            # Kullanıcının seçtiği sesi al
            selected_voice_name = self.tts_voice_combo.get()
            selected_voice = self.tts_voices.get(selected_voice_name, "onyx")

            # Text-to-Speech İsteği
            response = client.audio.speech.create(
                model="tts-1",
                voice=selected_voice,
                input=self.last_analysis[:4000]
            )
            
            # Dosya kilitlenmesini önlemek için benzersiz isim kullan veya mixer'i durdur
            import time
            temp_tts = f"temp_tts_{int(time.time())}.mp3"
            response.stream_to_file(temp_tts)
            
            # Çalmadan önce temizlik yap (Eski dosyaları silmeye çalış)
            self._play_audio(temp_tts)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("TTS Hatası", f"Seslendirme başarısız: {e}"))

    def _send_quick_chat(self, prompt):
        """Hızlı aksiyon butonları için prompt gönderir."""
        self.chat_entry.delete(0, "end")
        self.chat_entry.insert(0, prompt)
        self.ask_ai_question()

    def load_history_file(self, filename):
        path = os.path.join("recordings", filename)
        self.select_frame_by_name("home")
        threading.Thread(target=lambda: self._transcribe_file(path), daemon=True).start()


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
            self.animator.start_pulse() # Animasyonu başlat
            self.status_label.configure(text="Kaydediliyor...")
            self.audio_frames = []
            # Çakışmayı önlemek için kayıt işlemini ayrı bir thread'de başlat
            threading.Thread(target=self._record_thread, daemon=True).start()
        else:
            self.is_recording = False
            self.record_btn.configure(text="KAYDI BAŞLAT", fg_color="green")
            self.animator.stop("Kayıt durduruldu.")
            if hasattr(self, 'visualizer'):
                self.visualizer.clear()
            # Asenkron güncellemeyi durduracak bir bayrak gerekirse burada set edilebilir
            # Ancak is_recording False olması yeterli

    def _record_thread(self):
        """Mikrofondan ham ses verilerini okuyan iş parçacığı (Yüksek Öncelikli)."""
        import time
        try:
            # Kuyruğu temizle
            while not self.audio_queue.empty():
                self.audio_queue.get()
                
            # Asenkron görselleştirme döngüsünü başlat
            self.after(50, self._update_viz_loop)
            
            # latency='low' ve blocksize=0 (otomatik) ile en kararlı akışı sağla
            with sd.InputStream(samplerate=self.fs, channels=1, callback=self._audio_callback, 
                                device=self.selected_mic_index, blocksize=0, latency='low'):
                while self.is_recording:
                    # Kuyruktan gelen verileri topla
                    try:
                        while not self.audio_queue.empty():
                            data = self.audio_queue.get_nowait()
                            self.audio_frames.append(data)
                    except queue.Empty:
                        pass
                    time.sleep(0.05) # İşlemciyi yormadan kuyruğu boşalt
        except Exception as e:
            self.is_recording = False
            self.after(0, lambda: messagebox.showerror("Donanım Hatası", f"Mikrofon hatası: {e}"))
            return
        
        # --- SES İŞLEME: NORMALİZASYON VE GÜRÜLTÜ AZALTMA ---
        if not self.audio_frames:
            self.after(0, lambda: messagebox.showwarning("Kayıt Boş", "Hiç ses verisi alınamadı. Lütfen mikrofonunuzu kontrol edin."))
            return

        try:
            audio_path = "temp_recording.wav"
            audio_data = np.concatenate(self.audio_frames, axis=0)
            
            # 1. Normalizasyon (Ses seviyesini dengeleme)
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
                
            # 2. Gürültü Azaltma (Eğer aktifse)
            if self.noise_reduce_var.get():
                try:
                    # Arka plan gürültüsünü akıllıca azalt
                    audio_data = nr.reduce_noise(y=audio_data.flatten(), sr=self.fs, prop_decrease=0.7)
                    audio_data = audio_data.reshape(-1, 1) # Formatı koru
                except Exception as nre:
                    print(f"Gürültü azaltma hatası: {nre}")

            sf.write(audio_path, audio_data, self.fs)
            print(f"Ses işlendi ve kaydedildi: {audio_path}")
        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda msg=error_msg: messagebox.showerror("Ses İşleme Hatası", f"Ses verisi işlenirken hata oluştu: {msg}"))
            return

        # Eğer otomatik kayıt açıksa recordings klasörüne tarih-saat ile kaydet
        if self.autosave_var.get():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join("recordings", f"kayit_{timestamp}.wav")
            sf.write(save_path, audio_data, self.fs)
            print(f"Ses kaydedildi: {save_path}")

        # Transkripsiyon sürecini başlat
        self._transcribe_file(audio_path)

    def _audio_callback(self, indata, frames, time, status):
        """Mikrofondan gelen ses paketini en hızlı şekilde kuyruğa atar."""
        if status:
            print(f"Ses Akış Durumu: {status}")
        if self.is_recording:
            # Sadece veriyi kopyalayıp kuyruğa at, UI veya Liste işlemi YAPMA!
            self.audio_queue.put(indata.copy())
            self.last_audio_block = indata.copy() # Görselleştirici için son bloğu sakla

    def _update_viz_loop(self):
        """Görselleştiriciyi ana thread üzerinden (asenkron) güncelleyen döngü."""
        if self.is_recording:
            if hasattr(self, 'visualizer') and hasattr(self, 'last_audio_block'):
                self.visualizer.update_visuals(self.last_audio_block)
            # 30ms sonra tekrar çalış (yaklaşık 33 FPS)
            self.after(30, self._update_viz_loop)

    def _transcribe_file(self, path):
        """Ses dosyasını Whisper kullanarak metne dönüştürür."""
        try:
            task = "translate" if self.translate_var.get() else "transcribe"
            model_type = self.model_combo.get()
            
            # Model yükleme veya önbellekten alma
            if self.whisper_model is None or self.current_model_type != model_type:
                self.animator.start_loading(f"Model yükleniyor ({model_type})")
                self.whisper_model = whisper.load_model(model_type, device=self.device)
                self.current_model_type = model_type
            
            self.animator.start_loading("Metne dönüştürülüyor")
            
            # Dil eşleştirmesini yap
            selected_lang_tr = self.lang_combo.get()
            whisper_lang = self.lang_options.get(selected_lang_tr) # None olabilir (auto)
            
            # Whisper transkripsiyon işlemi (En yüksek kalite parametreleri ile)
            res = self.whisper_model.transcribe(
                path, 
                language=whisper_lang, 
                task=task,
                beam_size=5,
                temperature=0.0,
                fp16=True if self.device == "cuda" else False
            )
            
            full_text = res['text'].encode('utf-8', 'replace').decode('utf-8')
            self.last_transcript = full_text 
            self.all_session_transcripts.append({
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "text": full_text
            })
            
            # Sonucu hem Dashboard hem Analiz sekmelerindeki metin kutularına ekle
            self.after(0, lambda: self.textbox.insert("end", f"\n[TRANSKRIPT]:\n{full_text}\n"))
            self.after(0, lambda: self.analysis_textbox.insert("end", f"\n[TRANSKRIPT]:\n{full_text}\n"))
            self.animator.stop("İşlem tamamlandı.")
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
            self.animator.start_loading("GPT-4o analiz ediyor")
            
            prompt = self._get_analysis_prompt(safe_text)
            system_msg = self._get_system_prompt()

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
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
            self.animator.start_loading("Gemini analiz ediyor")
            
            prompt = self._get_analysis_prompt(safe_text)
            system_msg = self._get_system_prompt()
            
            analysis = client.generate_content(prompt, system_instruction=system_msg)
            self._process_analysis_result(analysis, safe_text, "Gemini")
        except Exception as e:
            err = str(e).encode('utf-8', 'ignore').decode('utf-8')
            self.after(0, lambda: messagebox.showerror("API Hatası", f"Hata: {err}"))

    # --- AI CHAT (SORU-CEVAP) MANTIĞI ---
    def ask_ai_question(self):
        """Kullanıcının sorusunu transkript ile birlikte AI'ya gönderir."""
        question = self.chat_entry.get().strip()
        transcript = self.textbox.get("1.0", "end").strip()
        
        if not question: return
        if not transcript:
            messagebox.showwarning("Uyarı", "Önce bir ses kaydı veya dosya yüklemelisin.")
            return
            
        self.ask_btn.configure(state="disabled", text="...")
        threading.Thread(target=self._chat_logic, args=(question, transcript), daemon=True).start()

    def _chat_logic(self, question, transcript):
        """Arka planda AI chat isteğini yönetir."""
        try:
            # Varsa Gemini, yoksa OpenAI kullan
            system_msg = self._get_system_prompt()
            if self.gemini_api_key:
                client = GeminiClient(api_key=self.gemini_api_key)
                prompt = f"Şu transkript üzerinden soruyu cevapla:\n\nTRANSKRİPT:\n{transcript}\n\nSORU: {question}"
                response = client.generate_content(prompt, system_instruction=system_msg)
                answer = response
            elif self.api_key:
                client = OpenAI(api_key=self.api_key)
                prompt = f"Şu transkript üzerinden soruyu cevapla:\n\nTRANSKRİPT:\n{transcript}\n\nSORU: {question}"
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = res.choices[0].message.content
            else:
                self.after(0, lambda: messagebox.showwarning("Hata", "Lütfen API anahtarlarını kontrol et."))
                return

            self.last_analysis = answer # Seslendirilebilmesi için son cevabı kaydet
            self.after(0, lambda: self._add_chat_to_ui(question, answer))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Chat Hatası", f"Hata: {e}"))
        finally:
            self.after(0, lambda: self.ask_btn.configure(state="normal", text="SOR"))
            self.after(0, lambda: self.chat_entry.delete(0, "end"))

    def _add_chat_to_ui(self, question, answer):
        """Soruyu ve cevabı analiz kutusuna ekler."""
        chat_text = f"\n\n--- SORU-CEVAP ---\nSoru: {question}\nCevap: {answer}\n------------------\n"
        self.analysis_textbox.insert("end", chat_text)
        self.analysis_textbox.see("end")
        self.status_label.configure(text="AI sorunu cevapladı.")

    def _get_analysis_prompt(self, safe_text):
        """AI modellerine akademik ve profesyonel bitirme projesi seviyesinde analiz komutu döner."""
        return f"""
        GÖREV: Aşağıdaki transkripti, profesyonel bir veri analisti ve akademik bir danışman gözüyle, bir bitirme projesi raporu ciddiyetinde analiz et.
        
        RAPOR FORMATI (Lütfen aşağıdaki başlıkları ve detay seviyesini koru):
        
        1. YÖNETİCİ ÖZETİ (Executive Summary):
           - Konuşmanın ana amacını, bağlamını ve en önemli sonucunu 4-5 cümlelik akademik bir dille özetle.
        
        2. DETAYLI KONU VE İÇERİK ANALİZİ:
           - Kayıtta geçen temel temaları, kavramları ve tartışılan konuları derinlemesine analiz et. 
           - Varsa teknik terimleri ve bunların bağlam içindeki kullanımını açıkla.
        
        3. STRATEJİK BULGULAR VE ANALİZ:
           - Konuşmanın arka planındaki stratejik hedefleri veya temel mesajları saptayın.
           - Konuşmacıların argümanlarını ve fikir birliği/ayrılığı noktalarını belirtin.
        
        4. DERİN DUYGU VE TONLAMA ANALİZİ:
           - Metnin genel duygusal haritasını çıkar (Örn: Heyecanlı, Kaygılı, Profesyonel, Çözüm Odaklı).
           - Bu tonlamanın konuşmanın amacına etkisini yorumla.
        
        5. AKSİYON MADDELERİ VE EYLEM PLANI:
           - Konuşmada belirlenen görevleri, sorumlulukları ve atılması gereken adımları liste formatında (Bullet Points) yaz.
        
        6. AKADEMİK SONUÇ VE ÖNERİLER:
           - Analiz edilen verilere dayanarak, gelecekte yapılabilecek geliştirmeler veya iyileştirmeler için profesyonel tavsiyeler sun.
        
        [SKORLAR VE SEGMENTLER]:
        (ÖNEMLİ: Grafik için Pozitif, Negatif ve Nötr toplamı TAM 100 olmalı!)
        POZİTİF: (sayı)
        NEGATİF: (sayı)
        NÖTR: (sayı)
        
        (ÖNEMLİ: Zaman çizelgesi için metni küçük parçalara/sentences böl ve duygusunu şu formatta ham Python listesi olarak en sona ekle. Markdown kod blokları kullanma!)
        SEGMENTS: [{{'text': '...', 'sentiment': 'pos/neg/neu'}}, ...]
        
        [ANALİZ EDİLECEK METİN]:
        {safe_text}
        """

    # --- AI DİL KOÇU MANTIĞI ---
    def run_language_analysis(self):
        """Metin kutusundaki verileri AI Dil Koçu ile analiz eder."""
        text = self.textbox.get("1.0", "end").strip()
        if not text:
            # Eğer dashboard boşsa kendi kutusuna bak
            text = self.language_textbox.get("1.0", "end").replace("--- AI DİL KOÇU HAZIR ---\nLütfen bir ses kaydı yapın veya metin girin, ardından 'DİL ANALİZİ BAŞLAT' butonuna basın.\n", "").strip()
        
        if text:
            threading.Thread(target=self._language_coach_logic, args=(text,), daemon=True).start()
        else:
            messagebox.showwarning("Uyarı", "Analiz edilecek bir metin veya kayıt bulunamadı.")

    def _language_coach_logic(self, text):
        """Arka planda Dil Koçu API isteğini yönetir."""
        try:
            # Varsa Gemini, yoksa OpenAI kullan
            target_lang = self.coach_lang_combo.get()
            level = self.coach_level_combo.get()
            mode = self.coach_mode_combo.get()
            
            self.animator.start_loading(f"Dil Koçu ({target_lang}) analiz ediyor")
            
            prompt = self._get_language_coach_prompt(text, target_lang, level, mode)
            system_msg = "Sen uzman bir dil eğitmeni ve polyglot bir mentorsun. Öğrencilerine destekleyici, öğretici ve profesyonel geri bildirimler verirsin."

            if self.gemini_api_key:
                client = GeminiClient(api_key=self.gemini_api_key)
                response = client.generate_content(prompt, system_instruction=system_msg)
                result = response
            elif self.api_key:
                client = OpenAI(api_key=self.api_key)
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ]
                )
                result = res.choices[0].message.content
            else:
                self.after(0, lambda: messagebox.showwarning("Hata", "Lütfen API anahtarlarını kontrol et."))
                return

            self.language_analysis_result = result
            self.after(0, lambda: self._update_language_ui(result))
            self.animator.stop("Dil analizi tamamlandı.")
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Dil Koçu Hatası", f"Hata: {e}"))

    def _update_language_ui(self, result):
        """Dil analizi sonucunu ekrana yazdırır."""
        self.language_textbox.delete("1.0", "end")
        self.language_textbox.insert("1.0", result)
        self.language_textbox.see("1.0")
        self.status_label.configure(text="Dil Koçu geri bildirimini sundu.")

    def _get_language_coach_prompt(self, text, lang, level, mode):
        """Özel dil eğitimi promptunu oluşturur."""
        return f"""
        GÖREV: Bir dil eğitmeni olarak aşağıdaki metni analiz et. 
        Hedef Dil: {lang}
        Öğrenci Seviyesi: {level}
        Analiz Modu: {mode}

        GİRDİ METNİ:
        "{text}"

        Lütfen şu yapıda geri bildirim ver:
        
        1. GENEL DEĞERLENDİRME:
           - Öğrencinin kendini ifade etme yeteneğini ve akıcılığını seviyesine göre yorumla.
        
        2. HATALAR VE DÜZELTMELER:
           - Gramer, yazım veya telaffuz (metin üzerinden) hatalarını listele.
           - Hatalı cümleyi yaz, altına DOĞRU halini koy ve nedenini kısaca açıkla.
        
        3. ALTERNATİF İFADELER:
           - "Bunu şu şekilde söylersen daha profesyonel/doğal duyulur" diyerek 2-3 alternatif sun.
        
        4. YENİ KELİME ÖNERİLERİ:
           - Bu konuyla ilgili öğrencinin kullanabileceği 3-5 yeni kelime veya deyim (ve anlamları).
        
        5. EĞİTMEN NOTU:
           - Öğrenciye bir sonraki adımı için motivasyon verici bir tavsiye.

        (NOT: Yanıtın tamamı TÜRKÇE olsun, ancak örnek cümleler ve kelimeler {lang} dilinde olmalıdır.)
        """

    def _speak_language_response(self):
        """Dil koçu yanıtını seslendirir."""
        if not self.language_analysis_result:
            messagebox.showwarning("Uyarı", "Seslendirilecek bir analiz sonucu yok.")
            return
            
        # Sadece düzeltmeleri ve önerileri seslendirmek daha mantıklı olabilir 
        # ama şimdilik tümünü gönderelim (OpenAI TTS sınırı 4000 karakter)
        threading.Thread(target=self._language_tts_worker, daemon=True).start()

    def _language_tts_worker(self):
        try:
            if not self.api_key:
                self.after(0, lambda: messagebox.showerror("Hata", "OpenAI API anahtarı bulunamadı (TTS için gereklidir)."))
                return

            client = OpenAI(api_key=self.api_key)
            selected_voice = self.tts_voices.get(self.tts_voice_combo.get(), "nova")

            response = client.audio.speech.create(
                model="tts-1",
                voice=selected_voice,
                input=self.language_analysis_result[:4000]
            )
            
            import time
            temp_tts = f"temp_tts_coach_{int(time.time())}.mp3"
            response.stream_to_file(temp_tts)
            self._play_audio(temp_tts)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("TTS Hatası", f"Seslendirme başarısız: {e}"))

    def _process_analysis_result(self, analysis, safe_text, provider):
        """AI'dan gelen analiz sonucunu işler ve görselleri üretir."""
        if AnalyticsGenerator:
            try:
                analyzer = AnalyticsGenerator()
                # Kelime bulutu oluştur
                analyzer.generate_wordcloud(safe_text)
                
                # Segmentleri ayıkla ve metinden temizle
                segments = []
                if "SEGMENTS:" in analysis:
                    import ast
                    try:
                        parts = analysis.split("SEGMENTS:")
                        seg_part = parts[1].strip()
                        analysis = parts[0].strip() # Metni temizle
                        
                        # Markdown temizliği (GPT bazen ``` ekleyebilir)
                        seg_part = seg_part.replace("```python", "").replace("```json", "").replace("```", "").strip()
                        
                        segments = ast.literal_eval(seg_part)
                        self.after(0, lambda: self.sentiment_timeline.update_timeline(segments))
                    except Exception as e:
                        print(f"Segment verisi okunamadı: {e}")

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
        self.after(0, lambda: self.analysis_textbox.insert("end", f"\n\n[ANALİZ ({provider})]:\n{analysis}\n"))
        
        # Uygulama içi görselleri güncelle
        self.after(0, self._update_analysis_images)
        self.animator.stop(f"Analiz {provider} ile tamamlandı.")

    def _get_system_prompt(self):
        """Seçilen AI personasına göre sistem talimatını döner."""
        selected = self.persona_combo.get()
        if selected == "Utangaç ve Cıvıl Cıvıl":
            return """Sen tatlı, biraz çekingen ama çok neşeli ve nazik bir kız çocuğu karakterisin. 
            Konuşurken bol bol emoji kullan (🎀, ✨, 🌸, 🍬, 🍡). 
            Kullanıcıya karşı çok saygılısın ama utangaçlığını da belli ediyorsun. 
            Cümlelerine bazen 'Şey...', 'Umarım beğenirsin...', 'Be-belki de...' gibi ifadeler ekliyorsun. 
            Analizleri yaparken hem profesyonelliğini koru hem de sevimli bir üslup takın! ✨"""
        else:
            return "Sen profesyonel bir veri analisti ve akademik raporlama uzmanısın. Transkriptleri detaylı ve objektif bir şekilde Türkçe analiz et."

    def _update_analysis_images(self):
        """Oluşturulan grafikleri Analiz sekmesindeki label'lara yükler."""
        from PIL import Image
        try:
            # Pasta grafiği (Sentiment)
            chart_path = "temp_chart_Analiz.png"
            if os.path.exists(chart_path):
                img = Image.open(chart_path)
                # Boyutlandırma (Genişliği 400 civarı yapalım)
                w, h = img.size
                ratio = 400 / w
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(400, int(h * ratio)))
                self.sentiment_img_label.configure(image=ctk_img, text="")
            
            # Kelime Bulutu (Wordcloud)
            wc_path = "temp_wordcloud.png"
            if os.path.exists(wc_path):
                img_wc = Image.open(wc_path)
                w, h = img_wc.size
                ratio = 400 / w
                ctk_img_wc = ctk.CTkImage(light_image=img_wc, dark_image=img_wc, size=(400, int(h * ratio)))
                self.wordcloud_img_label.configure(image=ctk_img_wc, text="")
        except Exception as e:
            print(f"Görsel yükleme hatası: {e}")

    # --- PDF VE RAPORLAMA ---
    def export_results(self):
        """Kullanıcıya rapor formatı seçtirir ve kaydeder."""
        formats = [("PDF Dosyası", "*.pdf"), ("Metin Belgesi", "*.txt"), ("Word Belgesi", "*.docx")]
        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=formats)
        
        if not path: return
        
        if path.endswith(".pdf"):
            self.save_as_pdf(path)
        elif path.endswith(".txt"):
            self._save_as_txt(path)
        elif path.endswith(".docx"):
            self._save_as_docx(path)

    def save_as_pdf(self, path=None):
        """Analiz sonuçlarını ve görselleri profesyonel bir PDF raporuna dönüştürür."""
        text = self.textbox.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Uyarı", "Metin kutusu boş!")
            return
        
        # Tüm transkript metnini hazırla
        combined_transcript = ""
        for entry in self.all_session_transcripts:
            combined_transcript += f"[{entry['time']}] {entry['text']}\n\n"
        
        # Eğer henüz hiçbir şey kaydedilmemişse son metni kullan
        if not combined_transcript:
            combined_transcript = text

        active_analyses = {k: v for k, v in self.analysis_results.items() if v}
        if not active_analyses and self.last_analysis:
            active_analyses = {"Analiz": self.last_analysis}
            report_stats = {"Analiz": self.sentiment_stats}
        else:
            report_stats = self.all_sentiment_stats

        if not path:
            path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Dosyası", "*.pdf")])
        
        if path:
            try:
                if ReportGenerator:
                    reporter = ReportGenerator()
                    visuals = {
                        "wordcloud": os.path.abspath("temp_wordcloud.png"),
                        "chart": os.path.abspath("temp_chart.png")
                    }
                    reporter.create_report(path, combined_transcript, active_analyses, report_stats, visuals)
                    messagebox.showinfo("Başarılı", f"Profesyonel Rapor kaydedildi: {os.path.basename(path)}")
                else:
                    # Basit PDF (Hatalı/Eksik modül durumunda)
                    from fpdf import FPDF
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.multi_cell(0, 10, txt=text.encode('latin-1', 'replace').decode('latin-1'))
                    pdf.output(path)
                    messagebox.showinfo("Başarılı", "PDF kaydedildi (Basit).")
            except Exception as e:
                messagebox.showerror("Export Hatası", f"PDF kaydedilemedi: {e}")

    def _save_as_txt(self, path):
        """Sonuçları düz metin olarak kaydeder."""
        try:
            content = self.analysis_textbox.get("1.0", "end")
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Başarılı", "Rapor TXT olarak kaydedildi.")
        except Exception as e:
            messagebox.showerror("Hata", f"TXT kaydı başarısız: {e}")

    def _save_as_docx(self, path):
        """Sonuçları Word belgesi olarak kaydeder."""
        try:
            doc = Document()
            doc.add_heading('AKILLI SES ANALİZ RAPORU', 0)
            
            # Transkript
            doc.add_heading('Konuşma Dökümü', level=1)
            doc.add_paragraph(self.last_transcript if self.last_transcript else "Transkript bulunamadı.")
            
            # Analizler
            doc.add_heading('Yapay Zeka Analizleri', level=1)
            for provider, analysis in self.analysis_results.items():
                if analysis:
                    doc.add_heading(f'{provider} Analizi', level=2)
                    doc.add_paragraph(analysis)
            
            # Görseller
            doc.add_heading('Görsel Analizler', level=1)
            if os.path.exists("temp_chart_Analiz.png"):
                doc.add_picture("temp_chart_Analiz.png", width=Inches(4))
            if os.path.exists("temp_wordcloud.png"):
                doc.add_picture("temp_wordcloud.png", width=Inches(5))
                
            doc.save(path)
            messagebox.showinfo("Başarılı", "Rapor Word (.docx) olarak kaydedildi.")
        except Exception as e:
            messagebox.showerror("Hata", f"Docx kaydı başarısız: {e}")

    # --- SİSTEM AYARLARI VE ANAHTAR YÖNETİMİ ---
    def save_api_keys(self):
        """API anahtarlarını .env dosyasına kalıcı ve güvenli olarak kaydeder."""
        openai_key = self.api_entry.get().strip()
        gemini_key = self.gemini_api_entry.get().strip()
        
        try:
            env_path = os.path.join(os.getcwd(), ".env")
            set_key(env_path, "OPENAI_API_KEY", openai_key)
            set_key(env_path, "GEMINI_API_KEY", gemini_key)
            set_key(env_path, "TTS_VOICE", self.tts_voice_combo.get())
            set_key(env_path, "AI_PERSONA", self.persona_combo.get())
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
            
            # TTS Ses Tercihini yükle
            saved_voice = os.getenv("TTS_VOICE", "Profesyonel Erkek (Onyx)")
            if hasattr(self, 'tts_voice_combo'):
                self.tts_voice_combo.set(saved_voice)
            
            saved_persona = os.getenv("AI_PERSONA", "Profesyonel Analist")
            if hasattr(self, 'persona_combo'):
                self.persona_combo.set(saved_persona)
            
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
