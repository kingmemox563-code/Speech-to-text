@echo off
setlocal

echo ==========================================
echo    SES ANALIZ SISTEMI - BASLATILIYOR
echo ==========================================

echo [1/3] Kutuphaneler kontrol ediliyor...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [!] Hata: Kutuphaneler yuklenemedi. Internet baglantinizi kontrol edin.
    pause
    exit /b
)

echo.
echo [2/3] Donanim kontrol ediliyor (FFmpeg)...
if not exist "ffmpeg.exe" (
    echo [!] ffmpeg.exe bulunamadi! Otomatik kurulum baslatiliyor...
    python setup_ffmpeg.py
) else (
    echo [+] FFmpeg hazir.
)

echo.
echo [3/3] Uygulama baslatiliyor...
python main.py
if %errorlevel% neq 0 (
    echo.
    echo [!] Uygulama beklenmedik bir sekilde sonlandi.
    pause
)

endlocal
