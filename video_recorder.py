import cv2
import os
from datetime import datetime
import time

def record_camera_stream(camera_id, rtsp_url, duration=120, output_folder=r"videos"):
    os.makedirs(output_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_folder, f"cam_{camera_id}_{timestamp}.mp4")

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"[{camera_id}] Error: Cannot open RTSP stream.")
        return None

    # Get FPS and size
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:  # fallback if camera doesn't report FPS
        fps = 20

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
    size = (width, height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, size)

    print(f"[{camera_id}] Recording started: {output_file}")
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            print(f"[{camera_id}] Failed to grab frame.")
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"[{camera_id}] Recording completed")
    return output_file
