import os
import json
from glob import glob

PROCESSED_DB_FILE = "processed_videos.json"
VIDEO_FOLDER = os.path.join(os.getcwd(), "videos")


def load_processed_db():
    if os.path.exists(PROCESSED_DB_FILE):
        with open(PROCESSED_DB_FILE, "r") as f:
            return set(json.load(f))
    return set()


def save_processed_db(processed_videos):
    with open(PROCESSED_DB_FILE, "w") as f:
        json.dump(list(processed_videos), f)

def get_next_video():
    processed_videos = load_processed_db()
    video_files = sorted(
        glob(os.path.join(VIDEO_FOLDER, "recording_*.mp4")),
        key=os.path.getmtime,
        reverse=False  # Process oldest first
    )

    for video in video_files:
        if video not in processed_videos:
            return video

    return None


def mark_video_as_processed(video_path):
    processed_videos = load_processed_db()
    processed_videos.add(video_path)
    save_processed_db(processed_videos)


def reset_processed_db():
    if os.path.exists(PROCESSED_DB_FILE):
        os.remove(PROCESSED_DB_FILE)
        print("Processed video DB has been reset.")


if __name__ == "__main__":
    video = get_next_video()
    if video:
        print(f"Latest unprocessed video: {video}")
        mark_video_as_processed(video)
        print("Marked as processed.")
    else:
        print("No unprocessed video found.")
