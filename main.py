"""
main.py - Uygulama Giriş Noktası
Uygulamanın başlatıldığı ana dosyadır. gui.py içindeki App sınıfını 
örnekleyerek kullanıcı arayüzünü döngüye sokar.
"""

from gui import App

if __name__ == "__main__":
    # gui.py içindeki App sınıfını (ana pencere) başlatır
    app = App()
    # Uygulamanın kapanana kadar açık kalmasını sağlayan ana döngü
    app.mainloop()
