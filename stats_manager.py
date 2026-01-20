import json
import os
from datetime import datetime

class StatsManager:
    """
    Kullanıcının öğrenme istatistiklerini yerel olarak yöneten sınıf.
    Quiz skorları, toplam çalışma süresi ve oturum sayılarını takip eder.
    """
    def __init__(self, filename="student_stats.json"):
        self.filename = filename
        self.stats = self._load_stats()

    def _load_stats(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Istatistik yükleme hatası: {e}")
        
        # Varsayılan değerler
        return {
            "total_sessions": 0,
            "total_words": 0,
            "learning_time_minutes": 0,
            "total_quizzes": 0,
            "average_quiz_score": 0,
            "last_active": "",
            "achievements": []
        }

    def save_stats(self):
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Istatistik kaydetme hatası: {e}")

    def add_session(self, words=0, minutes=0):
        self.stats["total_sessions"] += 1
        self.stats["total_words"] += words
        self.stats["learning_time_minutes"] += minutes
        self.stats["last_active"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_stats()

    def add_quiz_result(self, score):
        total_quizzes = self.stats.get("total_quizzes", 0)
        current_avg = self.stats.get("average_quiz_score", 0)
        
        # Yeni ortalamayı hesapla
        new_avg = ((current_avg * total_quizzes) + score) / (total_quizzes + 1)
        
        self.stats["total_quizzes"] = total_quizzes + 1
        self.stats["average_quiz_score"] = round(new_avg, 1)
        self.save_stats()

    def get_summary(self):
        return self.stats
