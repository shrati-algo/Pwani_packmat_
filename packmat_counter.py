import cv2
import numpy as np
import os
from ultralytics import YOLO
from datetime import datetime
import torch
import time
from collections import deque

# ---------------------------------------
# IOU calculation (FIXED)
# ---------------------------------------
def iou(b1, b2):
    x1, y1, x2, y2 = b1
    x1p, y1p, x2p, y2p = b2

    xi1 = max(x1, x1p)
    yi1 = max(y1, y1p)
    xi2 = min(x2, x2p)
    yi2 = min(y2, y2p)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2p - x1p) * (y2p - y1p)

    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0


# ---------------------------------------
# NMS (kept same but fixed logic)
# ---------------------------------------
def apply_nms(detections, iou_thresh=0.5):
    if not detections:
        return detections

    detections.sort(key=lambda x: x[2], reverse=True)
    filtered = []

    while detections:
        best = detections.pop(0)
        filtered.append(best)

        detections = [
            d for d in detections
            if iou(d[0], best[0]) < iou_thresh
        ]

    return filtered


# ---------------------------------------
# Tracker
# ---------------------------------------
class ObjectTracker:
    def __init__(self, iou_threshold=0.3, max_missed=5):
        self.tracks = {}
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_id = 0
        self.counted_ids = set()

    def update_tracks(self, detections, line_y, counter):
        updated_tracks = {}
        used_ids = set()

        for bbox, label, conf in detections:
            best_iou = 0
            best_id = None

            for obj_id, data in self.tracks.items():
                match_iou = iou(bbox, data["bbox"])
                if match_iou > best_iou and match_iou > self.iou_threshold:
                    if obj_id not in used_ids:
                        best_iou = match_iou
                        best_id = obj_id

            cy = (bbox[1] + bbox[3]) // 2

            print(f"[TRACKING] Object center Y: {cy}, Line Y: {line_y}")

            # ----------------------------
            # Existing object
            # ----------------------------
            if best_id is not None:
                last_y = self.tracks[best_id]["last_y"]

                updated_tracks[best_id] = {
                    "bbox": bbox,
                    "label": label,
                    "conf": conf,
                    "last_y": cy,
                    "missed": 0
                }
                if best_id not in self.counted_ids:
                    if last_y < line_y <= cy:  
                        counter += 1
                        print(f"COUNTED OBJECT {best_id}! New count = {counter}")
                        self.counted_ids.add(best_id)

                used_ids.add(best_id)

            # ----------------------------
            # New object
            # ----------------------------
            else:
                updated_tracks[self.next_id] = {
                    "bbox": bbox,
                    "label": label,
                    "conf": conf,
                    "last_y": cy,
                    "missed": 0
                }
                self.next_id += 1

        # Keep old unmatched tracks temporarily
        for obj_id, data in self.tracks.items():
            if obj_id not in used_ids:
                data["missed"] += 1
                if data["missed"] < self.max_missed:
                    updated_tracks[obj_id] = data

        self.tracks = updated_tracks
        return counter

class VideoProcessor:
    def __init__(self, video_path, model_path=r"packmat_i2.pt", camera_id=0, update_hook=None):


        self.cap = cv2.VideoCapture(video_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device}")

        self.model = YOLO(model_path).to(self.device)
        self.camera_id = camera_id
        self.update_hook = update_hook

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 20

        # Counting line
        self.line_y = int(self.frame_height * 0.75)
        self.line_start = (0, self.line_y)
        self.line_end = (self.frame_width, self.line_y)

        print(f"[INFO] Frame size: {self.frame_width}x{self.frame_height}, Line Y: {self.line_y}")

        self.counter = 0
        self.tracker = ObjectTracker()

        os.makedirs("outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = f"cam_{self.camera_id}_{timestamp}_output.mp4"
        output_path = os.path.join("outputs", output_filename)
        self.output_path = output_path

        print("from packmat:", output_path)

        self.target_size = (640, 480)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.target_size)

        # =============== NEW ===============
        self.buffer_seconds = 60
        max_buffer_frames = int(self.fps * self.buffer_seconds)
        self.frame_buffer = deque(maxlen=max_buffer_frames)
        # ===================================


    def process_video(self, stop_flag=None):
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open video stream.")

        while True:
            if stop_flag and stop_flag():
                print("processing stopped by user")
                break

            ret, frame = self.cap.read()
            if not ret:
                print("End of video.")
                break

            start_time = time.time()
            results = self.model(frame, conf=0.25, verbose=False, device=self.device)[0]

            detections = []
            inference_time = (time.time() - start_time) * 1000
            print(f"[GPU] Inference time: {inference_time:.2f} ms | Count: {self.counter}")

            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                conf = float(box.conf[0])

                if label.lower() in ["jerrycan_bundle", "carton", "carton_brown"] and conf > 0.6:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(((x1, y1, x2, y2), label, conf))

            detections = apply_nms(detections, iou_thresh=0.5)

            cv2.line(frame, self.line_start, self.line_end, (0, 0, 255), 2)

            self.counter = self.tracker.update_tracks(detections, self.line_y, self.counter)

            if self.update_hook:
                self.update_hook(self.counter)

            # Draw objects
            for obj_id, data in self.tracker.tracks.items():
                x1, y1, x2, y2 = data["bbox"]
                label = data["label"]
                conf = data["conf"]

                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Draw counter
            cv2.putText(frame, f"Counter: {self.counter}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            # ================= FIXED =================
            frame_resized = cv2.resize(frame, self.target_size)
            self.frame_buffer.append(frame_resized)

        self.cleanup()

        if self.update_hook:
            self.update_hook(self.counter)

        return self.counter, self.output_path


    def cleanup(self):
        self.cap.release()

        total_frames = len(self.frame_buffer)
        expected_frames = int(self.fps * self.buffer_seconds)

        print(f"[INFO] Total frames collected = {total_frames}")
        print(f"[INFO] Expected frames for {self.buffer_seconds}s = {expected_frames}")

        # If video is shorter → save full video
        if total_frames < expected_frames:
            print("[INFO] Saving full video (shorter than buffer).")
            frames_to_save = list(self.frame_buffer)
        else:
            print(f"[INFO] Saving last {self.buffer_seconds} seconds only.")
            frames_to_save = list(self.frame_buffer)[-expected_frames:]

        print(f"[INFO] Writing {len(frames_to_save)} frames to {self.output_path}")

        for f in frames_to_save:
            self.out.write(f)

        self.out.release()
        print("video closed successfully")


# def dummy_hook(count):
#     print("[HOOK] Live Count Updated:", count)

# if __name__ == "__main__":
#     video_path = r"C:\Users\shradha\Downloads\cam_1_2025-07-10_13-45-45.mp4"   # <-- put your video path here

#     vp = VideoProcessor(
#         video_path=video_path,
#         model_path=r"Pwani_packmat_-main\packmat_i2.pt",   # your YOLO model
#         camera_id=1,
#         update_hook=dummy_hook        # optional
#     )

#     count, output_path = vp.process_video()

#     print("\n===== FINAL RESULT =====")
#     print("Total Count :", count)
#     print("Output Video:", output_path)
 
