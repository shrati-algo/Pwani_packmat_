# (imports unchanged)
import os
import cv2
import time
from ultralytics import YOLO
from datetime import datetime
import torch
from collections import deque

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
# Merge overlapping detections
# ---------------------------------------
def merge_overlapping_detections(detections, merge_iou=0.35):
    if not detections:
        return []

    used = [False] * len(detections)
    merged = []

    for i in range(len(detections)):
        if used[i]:
            continue

        (x1, y1, x2, y2), label, conf = detections[i]
        group = [(x1, y1, x2, y2, conf)]
        used[i] = True

        changed = True
        while changed:
            changed = False

            gx1 = min(b[0] for b in group)
            gy1 = min(b[1] for b in group)
            gx2 = max(b[2] for b in group)
            gy2 = max(b[3] for b in group)
            gbox = (gx1, gy1, gx2, gy2)

            for j in range(len(detections)):
                if used[j]:
                    continue
                (bx1, by1, bx2, by2), blabel, bconf = detections[j]
                if blabel != label:
                    continue
                if iou(gbox, (bx1, by1, bx2, by2)) >= merge_iou:
                    group.append((bx1, by1, bx2, by2, bconf))
                    used[j] = True
                    changed = True

        mx1 = min(b[0] for b in group)
        my1 = min(b[1] for b in group)
        mx2 = max(b[2] for b in group)
        my2 = max(b[3] for b in group)
        mconf = max(b[4] for b in group)

        merged.append(((mx1, my1, mx2, my2), label, mconf))

    return merged


# ---------------------------------------
# Logger
# ---------------------------------------
class TinyLogger:
    def __init__(self, camera_id: int, log_dir="packmat_counter logs"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.path = os.path.join(log_dir, f"{ts}.log")
        self._fh = open(self.path, "w", encoding="utf-8", buffering=1)

    def log(self, *args):
        prefix = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(prefix, *args)
        self._fh.write(prefix + " " + " ".join(map(str, args)) + "\n")

    def close(self):
        self._fh.close()


# ---------------------------------------
# Tracker (unchanged)
# ---------------------------------------
class ObjectTracker:
    def __init__(self, iou_threshold=0.22, max_missed=5):
        self.tracks = {}
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_id = 0
        self.counted_ids = set()

    def update_tracks(self, detections, line_y, counter, logger=None):
        log = logger.log if logger else print
        updated_tracks = {}
        used_ids = set()

        for bbox, label, conf in detections:
            best_iou = 0
            best_id = None

            for obj_id, data in self.tracks.items():
                match_iou = iou(bbox, data["bbox"])
                if match_iou > best_iou and match_iou > self.iou_threshold:
                    best_iou = match_iou
                    best_id = obj_id

            cy = (bbox[1] + bbox[3]) // 2

            if best_id is not None:
                prev_state = self.tracks[best_id]["state"]

                updated_tracks[best_id] = {
                    "bbox": bbox,
                    "state": prev_state,
                    "missed": 0
                }
                used_ids.add(best_id)

                if best_id not in self.counted_ids:
                    if prev_state == "above" and cy >= line_y:
                        counter += 1
                        self.counted_ids.add(best_id)
                        updated_tracks[best_id]["state"] = "crossed"

            else:
                state = "above" if cy < line_y else "below"
                updated_tracks[self.next_id] = {
                    "bbox": bbox,
                    "state": state,
                    "missed": 0
                }
                self.next_id += 1

        for obj_id, data in self.tracks.items():
            if obj_id not in used_ids:
                data["missed"] += 1
                if data["missed"] < self.max_missed:
                    updated_tracks[obj_id] = data

        self.tracks = updated_tracks
        return counter


# ---------------------------------------
# Video Processor (NO FULL RECORDING)
# ---------------------------------------
# class VideoProcessor:
#     def __init__(self, model_path="packmat_i2.pt", camera_id=1, fps=20, buffer_seconds=120, frame_size=(1280, 720)):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model = YOLO(model_path).to(self.device)

#         self.fps = fps
#         self.buffer_seconds = buffer_seconds
#         self.fixed_total_frames = int(self.fps * self.buffer_seconds)

#         self.camera_id = camera_id
#         self.counter = 0
#         self.tracker = ObjectTracker()

#         self.target_size = frame_size
#         self.frame_buffer = deque(maxlen=self.fixed_total_frames * 3)

#         os.makedirs("outputs", exist_ok=True)
#         ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         self.output_path = f"outputs/cam_{camera_id}_{ts}_output.mp4"

#         self.line_y = None
#         self.logger = TinyLogger(camera_id)

#     def process_frame(self, frame):
#         if self.line_y is None:
#             self.line_y = int(frame.shape[0] * 0.7)

#         detections = []
#         results = self.model(frame, conf=0.25, verbose=False)[0]

#         for box in results.boxes:
#             label = self.model.names[int(box.cls[0])]
#             conf = float(box.conf[0])

#             if label.lower() in ["carton", "jerrycan_bundle"] and conf > 0.15:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 detections.append(((x1, y1, x2, y2), label, conf))

#         detections = apply_nms(detections)
#         detections = merge_overlapping_detections(detections)

#         self.counter = self.tracker.update_tracks(detections, self.line_y, self.counter)

#         resized = cv2.resize(frame, self.target_size)
#         self.frame_buffer.append((time.time(), resized))

#         return self.counter, self.output_path

#     def cleanup(self):
#         if not self.frame_buffer:
#             return

#         frames = [f for (_, f) in self.frame_buffer][-self.fixed_total_frames:]

#         writer = cv2.VideoWriter(
#             self.output_path,
#             cv2.VideoWriter_fourcc(*"mp4v"),
#             self.fps,
#             self.target_size
#         )

#         for f in frames:
#             writer.write(f)

#         writer.release()
#         self.logger.close()

class VideoProcessor:
    def __init__(
        self,
        model_path="packmat_i2.pt",
        camera_id=1,
        fps=20,
        buffer_seconds=120,
        frame_size=(1280, 720),
        initial_count=0,
        output_path=None
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path).to(self.device)

        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self.fixed_total_frames = int(self.fps * self.buffer_seconds)

        self.camera_id = camera_id
        self.counter = int(initial_count or 0)
        self.tracker = ObjectTracker()

        self.target_size = frame_size
        self.frame_buffer = deque(maxlen=self.fixed_total_frames * 3)

        os.makedirs("outputs", exist_ok=True)

        if output_path:
            self.output_path = output_path
        else:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.output_path = f"outputs/cam_{camera_id}_{ts}_output.mp4"

        self.line_y = None
        self.logger = TinyLogger(camera_id)

    def process_frame(self, frame):
        if self.line_y is None:
            self.line_y = int(frame.shape[0] * 0.7)

        detections = []
        results = self.model(frame, conf=0.25, verbose=False)[0]

        for box in results.boxes:
            label = self.model.names[int(box.cls[0])]
            conf = float(box.conf[0])

            if label.lower() in ["carton", "jerrycan_bundle"] and conf > 0.15:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(((x1, y1, x2, y2), label, conf))

        detections = apply_nms(detections)
        detections = merge_overlapping_detections(detections)

        self.counter = self.tracker.update_tracks(detections, self.line_y, self.counter)

        resized = cv2.resize(frame, self.target_size)
        self.frame_buffer.append((time.time(), resized))

        return self.counter, self.output_path

    def cleanup(self):
        if not self.frame_buffer:
            return

        frames = [f for (_, f) in self.frame_buffer][-self.fixed_total_frames:]

        writer = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            self.target_size
        )

        for f in frames:
            writer.write(f)

        writer.release()
        self.logger.close()