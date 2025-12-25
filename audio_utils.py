"""
audio_utils.py - Ses Donanım Yardımcı Araçları
Bu modül, sistemdeki ses giriş cihazlarını (mikrofonları) listelemek 
ve varsayılan cihazı bulmak için yardımcı fonksiyonlar içerir.
"""

import sounddevice as sd

def get_microphones():
    """
    Sistemde kayıt yapabilen kullanılabilir giriş cihazlarını listeler.
    
    Returns:
        list: "Cihazİndeksi: CihazAdı" formatında yaylı diziler listesi.
    """
    devices = sd.query_devices()
    # Sadece giriş kanalı (max_input_channels > 0) olan cihazları filtrele
    return [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_input_channels'] > 0]

def get_default_microphone_index():
    """
    Sistemdeki en uygun varsayılan mikrofonun indeksini bulmaya çalışır.
    
    Returns:
        int veya None: Bulunan mikrofonun indeksi veya bulunamazsa None.
    """
    devices = sd.query_devices()
    input_devs = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
    
    if not input_devs:
        return None
        
    try:
        # Öncelikle sitemin kendi belirlediği varsayılan giriş cihazını kontrol et
        default_in = sd.default.device[0]
        if default_in is not None and default_in >= 0:
             # Cihazın geçerli ve giriş kanalı olduğunu doğrula
             if devices[default_in]['max_input_channels'] > 0:
                  return default_in
    except:
        # Hata durumunda sessizce devam et
        pass
    
    # Varsayılan bulunamazsa, en çok giriş kanalına sahip olanı seç (genelde ana mikrofon budur)
    best = max(input_devs, key=lambda x: x[1]['max_input_channels'])
    return best[0]
