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
def apply_nms(detections, iou_thresh=0.7):
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
# Tiny logger (prints + plain text .log)
# ---------------------------------------
class TinyLogger:
    """
    Writes the exact same strings that you print to:
      1) console (print)
      2) a plain text .log file (opens in VS Code)

    Lowest space without compression headaches:
    - line-buffered writes
    - optional manual flush per line
    """
    def __init__(self, camera_id: int, log_dir="model_logs"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = os.path.join(log_dir, f"cam_{camera_id}_{ts}.log")
        # line-buffered text file (buffering=1)
        self._fh = open(self.path, "w", encoding="utf-8", buffering=1)

    def log(self, *args, sep=" ", end="\n"):
        msg = sep.join(str(a) for a in args) + end
        # keep printing as-is
        print(*args, sep=sep, end=end)
        # write to log
        self._fh.write(msg)
        # line-buffering handles most cases; keep flush for safety
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


# ---------------------------------------
# Tracker (FIXED)
# ---------------------------------------
class ObjectTracker:
    def __init__(self, iou_threshold=0.35, max_missed=2):
        self.tracks = {}
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_id = 0
        self.counted_ids = set()

    # ✅ ONLY ADDING logger PARAM, logic unchanged
    def update_tracks(self, detections, line_y, counter, logger=None):
        log = logger.log if logger is not None else print

        updated_tracks = {}
        used_ids = set()

        for bbox, label, conf in detections:
            best_iou = 0
            best_id = None

            # 🔑 Match only ACTIVE (not missed) tracks
            for obj_id, data in self.tracks.items():
                if data["missed"] > 0:
                    continue
                match_iou = iou(bbox, data["bbox"])
                if match_iou > best_iou and match_iou > self.iou_threshold:
                    best_iou = match_iou
                    best_id = obj_id

            # ✅ Use BOTTOM of bbox for conveyor counting
            cy = bbox[3]

            if best_id is not None:
                last_y = self.tracks[best_id]["last_y"]
                updated_tracks[best_id] = {
                    "bbox": bbox,
                    "label": label,
                    "conf": conf,
                    "last_y": cy,
                    "missed": 0
                }

                log(
                    f"ID: {best_id} | Label: {label} | Conf: {conf:.2f} | "
                    f"BBox: {bbox} | BottomY: {cy}"
                )

                # 🔥 COUNT LOGIC (ONCE PER OBJECT)
                if best_id not in self.counted_ids:
                    if last_y < line_y <= cy:
                        counter += 1
                        self.counted_ids.add(best_id)
                        log(
                            f"\n🚨 COUNT INCREMENTED 🚨\n"
                            f"Object ID {best_id} crossed line\n"
                            f"TOTAL COUNT = {counter}\n"
                        )

                        # ❌ RETIRE THIS TRACK IMMEDIATELY (no reuse)
                        updated_tracks.pop(best_id, None)
                        used_ids.add(best_id)
            else:
                updated_tracks[self.next_id] = {
                    "bbox": bbox,
                    "label": label,
                    "conf": conf,
                    "last_y": cy,
                    "missed": 0
                }
                log(
                    f"NEW ID: {self.next_id} | Label: {label} | Conf: {conf:.2f} | "
                    f"BBox: {bbox} | BottomY: {cy}"
                )
                self.next_id += 1

        # Handle missed tracks
        for obj_id, data in self.tracks.items():
            if obj_id not in used_ids:
                data["missed"] += 1
                if data["missed"] < self.max_missed and obj_id not in self.counted_ids:
                    updated_tracks[obj_id] = data

        self.tracks = updated_tracks
        return counter


# ---------------------------------------
# Video Processor (LOCKED 60.0s OUTPUT)
# ---------------------------------------
class VideoProcessor:
    def __init__(
        self,
        model_path="packmat_i2.pt",
        camera_id=1,
        frame_skip=1,
        fps=20,
        buffer_seconds=120,
        frame_size=(1280, 720)
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path).to(self.device)

        self.fps = int(fps)
        self.buffer_seconds = float(buffer_seconds)
        self.fixed_total_frames = int(round(self.fps * self.buffer_seconds))

        self.frame_skip = max(1, int(frame_skip))
        self.frame_index = 0
        self.camera_id = camera_id

        self.counter = 0
        self.tracker = ObjectTracker()

        self.target_size = tuple(frame_size)

        # Buffer (timestamp, frame)
        self.frame_buffer = deque(maxlen=self.fixed_total_frames * 3)

        os.makedirs("outputs", exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_path = f"outputs/cam_{camera_id}_{ts}_output.mp4"

        self.line_y = None

        # ✅ logger per camera instance (unique file)
        self.logger = TinyLogger(camera_id=camera_id, log_dir="model_logs")
        self.logger.log(f"[LOGGER] Started: {self.logger.path}")

    def process_frame(self, frame):
        self.frame_index += 1
        self.logger.log(f"\n[FRAME {self.frame_index}]")

        if not frame.flags.writeable:
            frame = frame.copy()

        if self.line_y is None:
            h = frame.shape[0]
            self.line_y = int(h * 0.75)

        detections = []

        if self.frame_index % self.frame_skip == 0:
            self.logger.log(f"Line Y position: {self.line_y}")

            results = self.model(frame, conf=0.15, verbose=False, device=self.device)[0]

            for box in results.boxes:
                cls = int(box.cls[0])
                label = self.model.names[cls]
                conf = float(box.conf[0])

                if label.lower() in ["carton", "carton_brown", "jerrycan_bundle","sack] and conf > 0.15:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(((x1, y1, x2, y2), label, conf))

            detections = apply_nms(detections)
            self.logger.log(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Detected Objects: {len(detections)}")

            # ✅ same tracker logic, just routed logs
            self.counter = self.tracker.update_tracks(
                detections, self.line_y, self.counter, logger=self.logger
            )

        self.logger.log(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CURRENT TOTAL COUNT = {self.counter} FOR CAMERA ID = {self.camera_id}"
        )

        # Draw line
        cv2.line(frame, (0, self.line_y), (frame.shape[1], self.line_y), (0, 0, 255), 2)

        # Draw tracked bboxes
        for data in self.tracker.tracks.values():
            x1, y1, x2, y2 = data["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put count text
        cv2.putText(
            frame,
            f"Count: {self.counter}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            2
        )

        resized = cv2.resize(frame, self.target_size)
        self.frame_buffer.append((time.time(), resized))

        return self.counter, self.output_path

    def cleanup(self):
        if not self.frame_buffer:
            self.logger.log("[CLEANUP] No frames in buffer; nothing to write.")
            self.logger.close()
            return

        now = time.time()
        start_t = now - self.buffer_seconds

        recent_frames = [f for (ts, f) in self.frame_buffer if ts >= start_t]

        if len(recent_frames) >= self.fixed_total_frames:
            frames_to_write = recent_frames[-self.fixed_total_frames:]
        else:
            frames_to_write = recent_frames[:]
            last = frames_to_write[-1] if frames_to_write else self.frame_buffer[-1][1]
            need = self.fixed_total_frames - len(frames_to_write)
            frames_to_write.extend([last] * need)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.target_size)

        for f in frames_to_write:
            writer.write(f)

        writer.release()
        self.logger.log(f"[SAVED] {self.output_path} | Duration locked to {self.buffer_seconds:.1f}s @ {self.fps} FPS")
        self.logger.log("[LOGGER] Closing log file.")
        self.logger.close()


