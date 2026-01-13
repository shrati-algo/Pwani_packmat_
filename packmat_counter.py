import cv2
import numpy as np
import os
from ultralytics import YOLO
from datetime import datetime
import torch
from collections import deque
from frame_Capture import CameraLoader


# ---------------------------------------
# IOU calculation
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
# NMS
# ---------------------------------------
def apply_nms(detections, iou_thresh=0.5):
    if not detections:
        return detections

    detections.sort(key=lambda x: x[2], reverse=True)
    filtered = []

    while detections:
        best = detections.pop(0)
        filtered.append(best)
        detections = [d for d in detections if iou(d[0], best[0]) < iou_thresh]

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

            if best_id is not None:
                last_y = self.tracks[best_id]["last_y"]

                updated_tracks[best_id] = {
                    "bbox": bbox,
                    "label": label,
                    "conf": conf,
                    "last_y": cy,
                    "missed": 0
                }

                print(
                    f"ID: {best_id} | Label: {label} | Conf: {conf:.2f} | "
                    f"BBox: {bbox} | CenterY: {cy}"
                )

                if best_id not in self.counted_ids:
                    if last_y < line_y <= cy:
                        counter += 1
                        self.counted_ids.add(best_id)
                        print(
                            f"🚨 Crossing detected! Object ID {best_id} crossed the line."
                        )
                        print(f"✅ Updated Count: {counter}")

                used_ids.add(best_id)

            else:
                updated_tracks[self.next_id] = {
                    "bbox": bbox,
                    "label": label,
                    "conf": conf,
                    "last_y": cy,
                    "missed": 0
                }

                print(
                    f"NEW ID: {self.next_id} | Label: {label} | Conf: {conf:.2f} | "
                    f"BBox: {bbox} | CenterY: {cy}"
                )

                self.next_id += 1

        for obj_id, data in self.tracks.items():
            if obj_id not in used_ids:
                data["missed"] += 1
                if data["missed"] < self.max_missed:
                    updated_tracks[obj_id] = data

        self.tracks = updated_tracks
        return counter


# ---------------------------------------
# Video Processor (LAST 60s ONLY)
# ---------------------------------------
class VideoProcessor:
    def __init__(
        self,
        model_path="packmat_i2.pt",
        camera_id=1,
        update_hook=None,
        frame_skip=2,
        fps=20,
        buffer_seconds=60,
        frame_size=(640, 480)
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path).to(self.device)

        self.fps = fps
        self.frame_skip = max(1, frame_skip)
        self.frame_index = 0

        self.counter = 0
        self.tracker = ObjectTracker()
        self.update_hook = update_hook

        self.target_size = frame_size
        self.buffer_seconds = buffer_seconds

        self.frame_buffer = deque(maxlen=int(self.fps * buffer_seconds))

        os.makedirs("outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = f"cam_{camera_id}_{timestamp}_output.mp4"
        self.output_path = os.path.join("outputs", output_filename)

        self.line_y = None
        self.line_start = None
        self.line_end = None

    # -------------------------------------------------
    def process_frame(self, frame):
        self.frame_index += 1

        print(f"\n[FRAME {self.frame_index}]")

        if frame is None:
            return self.counter, self.output_path

        if self.frame_index % self.frame_skip != 0:
            return self.counter, self.output_path

        if self.line_y is None:
            h, w = frame.shape[:2]
            self.line_y = int(h * 0.75)
            self.line_start = (0, self.line_y)
            self.line_end = (w, self.line_y)

        print(f"Line Y position: {self.line_y}")

        results = self.model(frame, conf=0.25, verbose=False, device=self.device)[0]

        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            label = self.model.names[cls]
            conf = float(box.conf[0])

            if label.lower() in ["jerrycan_bundle", "carton", "carton_brown"] and conf > 0.6:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(((x1, y1, x2, y2), label, conf))

        detections = apply_nms(detections)

        print(f"Detected Objects: {len(detections)}")

        if not detections:
            print("No objects detected in this frame.")
            print(f"Current Count: {self.counter}")

        cv2.line(frame, self.line_start, self.line_end, (0, 0, 255), 2)

        self.counter = self.tracker.update_tracks(
            detections, self.line_y, self.counter
        )

        if self.update_hook:
            self.update_hook(self.counter)

        for data in self.tracker.tracks.values():
            x1, y1, x2, y2 = data["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"Count: {self.counter}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            2
        )

        frame_resized = cv2.resize(frame, self.target_size)

        self.frame_buffer.append(frame_resized)

        self._rewrite_last_60s_video()

        return self.counter, self.output_path

    # -------------------------------------------------
    def _rewrite_last_60s_video(self):
        if not self.frame_buffer:
            return

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            self.target_size
        )

        for frame in self.frame_buffer:
            writer.write(frame)

        writer.release()

    def cleanup(self):
        pass
