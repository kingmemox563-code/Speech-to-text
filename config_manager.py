# AYAR YÖNETİCİSİ (CONFIG MANAGER)
# Bu sınıf, kullanıcının seçtiği ayarları (mikrofon, dil, model vb.) kaydeder.
# Program kapatılıp açıldığında bu ayarları hatırlar.

import json
import os

CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "mic_index": None,
    "model_size": "large", # Kullanıcının RTX 3060 kartı var, Gemini kalitesi için Large şart.
    "language": "turkish",
    "theme": "Dark",
    "push_to_talk": False,
    "translate_mode": False,
    "api_key": ""
}

class ConfigManager:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        if not os.path.exists(CONFIG_FILE):
            return DEFAULT_CONFIG.copy()
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Config loading error: {e}")
            return DEFAULT_CONFIG.copy()

    def save_config(self, key, value):
        self.config[key] = value
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Config saving error: {e}")

    def get(self, key):
        return self.config.get(key, DEFAULT_CONFIG.get(key))
