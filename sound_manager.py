import os
import pygame

class SoundManager:
    """Arka plan atmosfer seslerini yönetir."""
    def __init__(self):
        self.current_sound = None
        self.is_playing = False
        
        # Klasör kontrolü
        if not os.path.exists("sounds"):
            os.makedirs("sounds")
            
    def play_ambience(self, scenario_name):
        """Senaryoya uygun sesi bulur ve döngüye alır."""
        # Senaryo adına göre dosya ara (örn: "Uzay Kolonisi" -> "uzay.mp3" veya "colony.mp3")
        # Basit eşleştirme: Dosya adında senaryodan bir kelime geçiyor mu?
        
        target_file = None
        
        # Mapping (Eşleştirme)
        mapping = {
            "Uzay": "space.mp3",
            "Mars": "space.mp3",
            "Zombi": "zombie.mp3",
            "Kıyamet": "zombie.mp3",
            "Orta Çağ": "medieval.mp3",
            "Tarih": "medieval.mp3",
            "Detektif": "noir.mp3",
            "Gizem": "noir.mp3",
            "Orman": "forest.mp3",
            "Doğa": "forest.mp3"
        }
        
        found_key = next((k for k in mapping if k in scenario_name), None)
        if found_key:
            target_file = mapping[found_key]
        
        if target_file:
            full_path = os.path.join("sounds", target_file)
            if os.path.exists(full_path):
                self._fade_and_play(full_path)
            else:
                print(f"Ambiyans dosyası bulunamadı: {full_path}")
                self.stop_ambience()
        else:
            self.stop_ambience()

    def _fade_and_play(self, path):
        if self.current_sound == path and self.is_playing:
            return

        try:
            pygame.mixer.music.fadeout(1000)
            pygame.mixer.music.load(path)
            pygame.mixer.music.play(loops=-1, fade_ms=1000)
            pygame.mixer.music.set_volume(0.3) # %30 ses seviyesi
            self.current_sound = path
            self.is_playing = True
        except Exception as e:
            print(f"Ses çalma hatası: {e}")

    def stop_ambience(self):
        pygame.mixer.music.fadeout(1000)
        self.current_sound = None
        self.is_playing = False
