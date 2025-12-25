"""
gemini_client.py - Google Gemini API İstemcisi
Bu modül, Google'ın Gemini Vision/Pro modellerine metin tabanlı 
analiz istekleri göndermek için kullanılır.
"""

import google.generativeai as genai
import os

class GeminiClient:
    """
    Google Gemini API ile etkileşimi yöneten sınıf.
    """
    def __init__(self, api_key):
        """
        İstemciyi yapılandırır ve modeli başlatır.
        """
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        # Varsayılan model (Hesabınızda en stabil görünen)
        self.model_name = 'gemini-2.5-flash' 
        self.model = genai.GenerativeModel(self.model_name)

    def generate_content(self, prompt, system_instruction=None):
        """
        Verilen isteme (prompt) dayanarak yapay zeka yanıtı oluşturur.
        [V19 Versiyonu]
        """
        try:
            if system_instruction:
                full_prompt = f"SYSTEM: {system_instruction}\n\nUSER: {prompt}"
            else:
                full_prompt = prompt
                
            try:
                # Aktif modeli dene
                print(f"[V19] Deneniyor: {self.model_name}")
                response = self.model.generate_content(full_prompt)
                return response.text
            except Exception as e:
                # 404 veya model hatası durumunda gerçek yedekleri dene
                print(f"[V19] {self.model_name} hatası: {e}. Yedek can simidi başlatılıyor...")
                
                # Sizin API hesabınızda ListModels ile doğrulanmış modeller:
                fallbacks = [
                    'gemini-flash-latest', 
                    'gemini-pro-latest',
                    'gemini-2.5-flash',
                    'gemini-2.0-flash'
                ]
                
                for fallback in fallbacks:
                    if fallback == self.model_name:
                        continue
                    try:
                        print(f"[V19] Yedek deneniyor: {fallback}")
                        temp_model = genai.GenerativeModel(fallback)
                        response = temp_model.generate_content(full_prompt)
                        self.model = temp_model 
                        self.model_name = fallback
                        return response.text
                    except Exception as fe:
                        print(f"[V19] {fallback} başarısız: {fe}")
                        continue
                
                return f"[V19] Gemini Hatası: Modellerin hiçbiri yanıt vermedi. Lütfen API anahtarınızı (AIza...) ve internetinizi kontrol edin.\nDenenenler: {self.model_name}, {', '.join(fallbacks)}\nSon Hata: {str(e)}"
                
        except Exception as e:
            return f"[V19] Beklenmedik Hata: {str(e)}"
