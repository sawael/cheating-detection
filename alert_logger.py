import pandas as pd
from datetime import datetime

video_log_file = "cheating_log.csv"
audio_log_file = "audio_log.csv"

for f in [video_log_file, audio_log_file]:
    pd.DataFrame(columns=["Time", "Event"]).to_csv(f, index=False)

def log_event(file, event):
    """Write a new alert line to a given CSV file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[timestamp, event]], columns=["Time", "Event"])
    df.to_csv(file, mode='a', header=False, index=False)
    print(f"[{timestamp}] {event}")