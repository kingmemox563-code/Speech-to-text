import os
import zipfile
import shutil
import urllib.request
import sys

# Resmi güncel release linki
URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
ZIP_NAME = "ffmpeg_temp.zip"

def download_file(url, filename):
    print(f"[*] FFmpeg indiriliyor: {url}")
    print("[*] Bu işlem internet hızınıza bağlı olarak birkaç dakika sürebilir...")
    
    # User-Agent eklemek bazı sunucularda engellenmeyi önler
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
    print("[*] Dosyalar ayıklanıyor...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # zip içindeki ffmpeg.exe dosyasını bul
            ffmpeg_exe_in_zip = None
            for file in zip_ref.namelist():
                if file.endswith("bin/ffmpeg.exe"):
                    ffmpeg_exe_in_zip = file
                    break
            
            if not ffmpeg_exe_in_zip:
                print("[-] Hata: Zip dosyası içinde ffmpeg.exe bulunamadı.")
                return False

            # Sadece ffmpeg.exe'yi çıkar
            zip_ref.extract(ffmpeg_exe_in_zip, ".")
            
            # Çıkarılan dosyayı ana dizine taşı
            source_path = os.path.normpath(ffmpeg_exe_in_zip)
            dest_path = "ffmpeg.exe"
            
            if os.path.exists(dest_path):
                os.remove(dest_path)
            
            shutil.move(source_path, dest_path)
            
            # Gereksiz klasör yapısını temizle
            top_dir = source_path.split(os.sep)[0]
            if os.path.exists(top_dir):
                shutil.rmtree(top_dir)
                
            print(f"[+] FFmpeg başarıyla ana dizine taşındı: {os.path.abspath(dest_path)}")
            return True
    except Exception as e:
        print(f"[-] Ayıklama hatası: {e}")
        return False

def setup():
    """Ana kurulum fonksiyonu"""
    if os.path.exists("ffmpeg.exe"):
        print("[!] FFmpeg zaten mevcut, kurulum atlanıyor.")
        return

    try:
        download_file(URL, ZIP_NAME)
        if extract_ffmpeg(ZIP_NAME):
            print("\n[SUCCESS] Kurulum başarıyla tamamlandı.")
        
        # Geçici zip dosyasını temizle
        if os.path.exists(ZIP_NAME):
            os.remove(ZIP_NAME)
            
    except Exception as e:
        print(f"\n[FAILED] Kurulum sırasında hata oluştu: {e}")

if __name__ == "__main__":
    setup()
