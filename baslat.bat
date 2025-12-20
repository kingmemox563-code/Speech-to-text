@echo off
echo Gerekli kutuphaneler kontrol ediliyor (Internet baglantisi gerekir)...
pip install -r requirements.txt
echo.
echo Baslatiliyor...
python main.py
if errorlevel 1 pause
