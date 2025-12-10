import webrtcvad
import sounddevice as sd
import numpy as np
import threading
import time
from alert_logger import log_event, audio_log_file

vad = webrtcvad.Vad(2)  # Sensitivity: 0–3
sample_rate = 16000
frame_duration = 30  # ms
frame_size = int(sample_rate * frame_duration / 1000)

def detect_talking():
    """Continuously detect voice activity in the background."""
    print("Audio monitoring started...")
    while True:
        try:
            audio = sd.rec(int(1 * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
            sd.wait()
            audio = audio.flatten()
            talking = False

            for start in range(0, len(audio), frame_size):
                frame = audio[start:start + frame_size]
                if len(frame) < frame_size:
                    break
                if vad.is_speech(frame.tobytes(), sample_rate):
                    talking = True
                    break

            if talking:
                log_event(audio_log_file, "Suspicious: Talking detected")

            time.sleep(0.5)
        except KeyboardInterrupt:
            print("Audio detection stopped.")
            break

def start_audio_thread():
    """Run audio detection in a background thread."""
    thread = threading.Thread(target=detect_talking, daemon=True)
    thread.start()
    return thread