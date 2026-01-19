import os
import time
from elevenlabs.client import ElevenLabs
from elevenlabs import save, VoiceSettings

class ElevenLabsManager:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.client = None
        if api_key:
            self.client = ElevenLabs(api_key=api_key)

    def update_key(self, api_key):
        self.api_key = api_key
        self.client = ElevenLabs(api_key=api_key)

    def get_voices(self):
        """Kullanıcının hesabındaki tüm sesleri döner."""
        if not self.client:
            return []
        try:
            voices_response = self.client.voices.get_all()
            # [voice_name, voice_id] listesi döner
            return [[v.name, v.voice_id] for v in voices_response.voices]
        except Exception as e:
            print(f"ElevenLabs Voices Error: {e}")
            return []

    def generate_speech(self, text, voice_id, model_id="eleven_multilingual_v2"):
        """Belirli bir ses ID'si ile metni seslendirir."""
        if not self.client or not voice_id:
            return None
        
        try:
            # v2.x SDK için convert metodu kullanılır
            audio_iterator = self.client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id=model_id,
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.75,
                    style=0.0,
                    use_speaker_boost=True
                )
            )
            
            output_path = f"temp_eleven_{int(time.time())}.mp3"
            save(audio_iterator, output_path)
            return output_path
        except Exception as e:
            print(f"ElevenLabs Generate Error: {e}")
            return None
