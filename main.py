from video_module import run_video_detection
from audio_module import start_audio_thread

if __name__ == "__main__":
    print("Initializing Deep Learning Cheating Detection System...")
    start_audio_thread()
    run_video_detection()