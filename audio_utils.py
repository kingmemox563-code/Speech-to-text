import sounddevice as sd

def get_microphones():
    """Returns a list of available input devices."""
    devices = sd.query_devices()
    return [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_input_channels'] > 0]

def get_default_microphone_index():
    """Attempts to find the best default microphone."""
    devices = sd.query_devices()
    input_devs = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
    if not input_devs:
        return None
    # Simple heuristic: pick the one with most input channels or just the system default
    # Usually sounddevice puts the default device at a specific index, but iterating is safer
    try:
        # Try to find default input device
        default_in = sd.default.device[0]
        if default_in is not None and default_in >= 0:
             # verify it's valid
             if devices[default_in]['max_input_channels'] > 0:
                 return default_in
    except:
        pass
    
    # Fallback to max channels
    best = max(input_devs, key=lambda x: x[1]['max_input_channels'])
    return best[0]
