import time
from video_recorder import record_camera_stream
from get_rtsp_link import get_rtsp_link
from packmat_counter_2 import VideoProcessor
from video_tracker import mark_video_as_processed
import os

def process_camera(camera_id, duration=120):
    print(f"Starting process for camera ID: {camera_id}")

    # Get RTSP link
    rtsp_url = get_rtsp_link(camera_id)
    if not rtsp_url:
        print(f"[{camera_id}] Error: RTSP link not found.")
        return

    # Step 1: record video
    recorded_path = record_camera_stream(camera_id, rtsp_url, duration=duration)
    if not recorded_path or not os.path.exists(recorded_path):
        print(f"[{camera_id}] Recording failed or file not found.")
        return

    # Step 2: Process with inference model
    print(f"[{camera_id}] Starting model inference on: {recorded_path}")
    processor = VideoProcessor(video_path=recorded_path, model_path=r"packmat_model.pt")
    processor.process_video()  # Saves output to /output folder

    # Step 3: Mark as processed
    mark_video_as_processed(recorded_path)
    print(f"[{camera_id}] Video processed and marked as done.")

if __name__ == "__main__":
    # Example: Run once for a single camera
    process_camera(camera_id="117")
