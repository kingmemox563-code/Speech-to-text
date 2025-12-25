"""
config_manager.py - Yapılandırma ve Ayar Yönetimi Modülü
Bu modül, uygulamanın ayarlarını (tema, dil, model boyutu vb.) bir JSON dosyasında 
saklamak ve .env dosyasından API anahtarlarını okumak için kullanılır.
"""

import json
import os
from dotenv import load_dotenv

# .env dosyasındaki çevresel değişkenleri yükle (API anahtarları vb.)
load_dotenv()

CONFIG_FILE = "config.json"

# Uygulama ilk kez çalıştığında veya ayar bulunamadığında kullanılacak varsayılan değerler
DEFAULT_CONFIG = {
    "mic_index": None,
    "model_size": "large",
    "language": "turkish",
    "theme": "Dark",
    "push_to_talk": False,
    "translate_mode": False
}

class ConfigManager:
    """
    Uygulama ayarlarını okuma ve yazma işlemlerini merkezi olarak yöneten sınıf.
    """
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        """
        config.json dosyasından ayarları yükler. Dosya yoksa varsayılanları döner.
        
        Returns:
            dict: Uygulama ayarları sözlüğü.
        """
        if not os.path.exists(CONFIG_FILE):
            return DEFAULT_CONFIG.copy()
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Konfigürasyon yükleme hatası: {e}")
            return DEFAULT_CONFIG.copy()

    def save_config(self, key, value):
        """
        Belirli bir ayarı günceller ve config.json dosyasına kaydeder.
        
        Args:
            key (str): Ayar anahtarı.
            value (any): Ayar değeri.
        """
        self.config[key] = value
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Konfigürasyon kaydetme hatası: {e}")

    def get(self, key):
        """
        İstenen ayar değerini döner. API anahtarları için öncelikle .env dosyasına bakar.
        
        Args:
            key (str): İstenen ayar veya API anahtarı.
            
        Returns:
            any: Bulunan değer veya varsayılan değer.
        """
        # API anahtarları güvenlik nedeniyle config.json yerine .env dosyasında saklanır
        if key == "openai_api_key":
            return os.getenv("OPENAI_API_KEY")
        if key == "gemini_api_key":
            return os.getenv("GEMINI_API_KEY")
        
        # Diğer genel ayarlar için config sözlüğüne bak
        return self.config.get(key, DEFAULT_CONFIG.get(key))
