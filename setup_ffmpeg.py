"""
setup_ffmpeg.py - FFmpeg Kurulum Yardımcısı
Bu betik, uygulama için gerekli olan FFmpeg kütüphanesini otomatik olarak 
indirir, ayıklar ve ana dizine yerleştirir.
"""

import os
import zipfile
import shutil
import urllib.request
import sys

# Resmi güncel FFmpeg release linki
URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
ZIP_NAME = "ffmpeg_temp.zip"

def download_file(url, filename):
    """FFmpeg zip dosyasını internetten indirir."""
    print(f"[*] FFmpeg indiriliyor: {url}")
    print("[*] Bu işlem internet hızınıza bağlı olarak birkaç dakika sürebilir...")
    
    # Sunucu tarafından engellenmemek için User-Agent başlığı ekle
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("[+] İndirme tamamlandı.")
    except Exception as e:
        print(f"[-] İndirme hatası: {e}")
        raise

def extract_ffmpeg(zip_path):
    """İndirilen zip dosyasından sadece gerekli olan ffmpeg.exe'yi ayıklar."""
    print("[*] Dosyalar ayıklanıyor...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Zip içindeki tüm dosyaları tara ve ffmpeg.exe'yi bul
            ffmpeg_exe_in_zip = None
            for file in zip_ref.namelist():
                if file.endswith("bin/ffmpeg.exe"):
                    ffmpeg_exe_in_zip = file
                    break
            
            if not ffmpeg_exe_in_zip:
                print("[-] Hata: Zip dosyası içinde ffmpeg.exe bulunamadı.")
                return False

            # Sadece ffmpeg.exe'yi geçici olarak çıkar
            zip_ref.extract(ffmpeg_exe_in_zip, ".")
            
            # Çıkarılan dosyayı projenin ana dizinine taşı
            source_path = os.path.normpath(ffmpeg_exe_in_zip)
            dest_path = "ffmpeg.exe"
            
            if os.path.exists(dest_path):
                os.remove(dest_path)
            
            shutil.move(source_path, dest_path)
            
            # Çıkarma sırasında oluşan gereksiz klasör yapısını temizle
            top_dir = source_path.split(os.sep)[0]
            if os.path.exists(top_dir):
                shutil.rmtree(top_dir)
                
            print(f"[+] FFmpeg başarıyla ana dizine taşındı: {os.path.abspath(dest_path)}")
            return True
    except Exception as e:
        print(f"[-] Ayıklama hatası: {e}")
        return False

def setup():
    """Tüm kurulum sürecini (indirme ve ayıklama) yöneten ana fonksiyon."""
    # Eğer dosya zaten varsa işlemi atla
    if os.path.exists("ffmpeg.exe"):
        print("[!] FFmpeg zaten mevcut, kurulum atlanıyor.")
        return

    try:
        # 1. Dosyayı indir
        download_file(URL, ZIP_NAME)
        # 2. Dosyayı ayıkla ve yerleştir
        if extract_ffmpeg(ZIP_NAME):
            print("\n[BAŞARILI] Kurulum başarıyla tamamlandı.")
        
        # 3. İndirilen geçici zip dosyasını temizle
        if os.path.exists(ZIP_NAME):
            os.remove(ZIP_NAME)
            
    except Exception as e:
        print(f"\n[BAŞARISIZ] Kurulum sırasında hata oluştu: {e}")

if __name__ == "__main__":
    setup()
