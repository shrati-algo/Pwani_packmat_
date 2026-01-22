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
# Merge overlapping detections (single box)
# ---------------------------------------
def merge_overlapping_detections(detections, merge_iou=0.35):
    """
    detections: [((x1,y1,x2,y2), label, conf), ...]
    Merges overlapping boxes (same label) into one bigger box.
    merge_iou fixed to 0.20 (as requested).
    """
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
# Tiny logger (prints + plain text .log)
# ---------------------------------------
class TinyLogger:
    def __init__(self, camera_id: int, log_dir="model_logs"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = os.path.join(log_dir, f"cam_{camera_id}_{ts}.log")
        self._fh = open(self.path, "w", encoding="utf-8", buffering=1)

    def log(self, *args, sep=" ", end="\n"):
        msg = sep.join(str(a) for a in args) + end
        print(*args, sep=sep, end=end)
        self._fh.write(msg)
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
    def __init__(self, iou_threshold=0.3, max_missed=4):
        self.tracks = {}
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_id = 0
        self.counted_ids = set()

    def update_tracks(self, detections, line_y, counter, logger=None):
        log = logger.log if logger is not None else print

        updated_tracks = {}
        used_ids = set()

        for bbox, label, conf in detections:
            best_iou = 0
            best_id = None

            # ✅ CHANGE 1: allow matching to tracks even if missed (until max_missed)
            for obj_id, data in self.tracks.items():
                if data.get("missed", 0) >= self.max_missed:
                    continue

                match_iou = iou(bbox, data["bbox"])
                if match_iou > best_iou and match_iou > self.iou_threshold:
                    best_iou = match_iou
                    best_id = obj_id

            cy = bbox[3]  # bottom y

            if best_id is not None:
                last_y = self.tracks[best_id]["last_y"]
                updated_tracks[best_id] = {
                    "bbox": bbox,
                    "label": label,
                    "conf": conf,
                    "last_y": cy,
                    "missed": 0
                }
                used_ids.add(best_id)

                log(
                    f"ID: {best_id} | Label: {label} | Conf: {conf:.2f} | "
                    f"BBox: {bbox} | BottomY: {cy}"
                )

                # ✅ CHANGE 2: robust line crossing using a small band
                band = 6
                if best_id not in self.counted_ids:
                    if last_y < (line_y - band) and cy >= (line_y + band):
                        counter += 1
                        self.counted_ids.add(best_id)
                        log(
                            f"\n🚨 COUNT INCREMENTED 🚨\n"
                            f"Object ID {best_id} crossed line\n"
                            f"TOTAL COUNT = {counter}\n"
                        )

                        updated_tracks.pop(best_id, None)

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
            if obj_id in used_ids:
                continue
            data["missed"] = data.get("missed", 0) + 1
            if data["missed"] < self.max_missed and obj_id not in self.counted_ids:
                updated_tracks[obj_id] = data

        self.tracks = updated_tracks
        return counter


# ---------------------------------------
# Video Processor (120s clip + FULL recording)
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

        # Buffer (timestamp, frame) for 120s clip
        self.frame_buffer = deque(maxlen=self.fixed_total_frames * 3)

        os.makedirs("outputs", exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_path = f"outputs/cam_{camera_id}_{ts}_output.mp4"

        # ✅ FULL video path: same name + "_full"
        base, ext = os.path.splitext(self.output_path)
        self.full_output_path = f"{base}_full{ext}"

        self.line_y = None

        # ✅ logger per camera instance (unique file)
        self.logger = TinyLogger(camera_id=camera_id, log_dir="model_logs")
        self.logger.log(f"[LOGGER] Started: {self.logger.path}")

        # ✅ full writer (lazy init on first frame)
        self._full_writer = None
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    def _init_full_writer_if_needed(self):
        if self._full_writer is not None:
            return
        self._full_writer = cv2.VideoWriter(
            self.full_output_path,
            self._fourcc,
            self.fps,
            self.target_size
        )
        if not self._full_writer.isOpened():
            self.logger.log(f"[ERROR] Could not open FULL writer: {self.full_output_path}")
            self._full_writer = None
        else:
            self.logger.log(f"[FULL] Recording started: {self.full_output_path}")

    def process_frame(self, frame):
        self.frame_index_client = getattr(self, "frame_index_client", 0) + 1
        self.frame_index = self.frame_index_client

        self.logger.log(f"\n[FRAME {self.frame_index}]")

        if not frame.flags.writeable:
            frame = frame.copy()

        if self.line_y is None:
            h = frame.shape[0]
            self.line_y = int(h * 0.75)

        detections = []

        if self.frame_index % self.frame_skip == 0:
            self.logger.log(f"Line Y position: {self.line_y}")

            results = self.model(frame, conf=0.25, verbose=False, device=self.device)[0]

            for box in results.boxes:
                cls = int(box.cls[0])
                label = self.model.names[cls]
                conf = float(box.conf[0])

                if label.lower() in ["carton", "carton_brown", "jerrycan_bundle", "sack"] and conf > 0.15:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(((x1, y1, x2, y2), label, conf))

            # ✅ CHANGE 3: NMS with 0.5 (already default)
            detections = apply_nms(detections, iou_thresh=0.5)

            # ✅ CHANGE 4: MERGE overlapping boxes into ONE (merge_iou fixed to 0.20)
            detections = merge_overlapping_detections(detections, merge_iou=0.20)

            self.logger.log(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Detected Objects: {len(detections)}")

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

        # ✅ resize ONCE (used for both: buffer clip + full video)
        resized = cv2.resize(frame, self.target_size)

        # ✅ keep 120s buffer behavior exactly the same
        self.frame_buffer.append((time.time(), resized))

        # ✅ write full annotated video continuously
        self._init_full_writer_if_needed()
        if self._full_writer is not None:
            self._full_writer.write(resized)

        return self.counter, self.output_path  # unchanged signature

    def cleanup(self):
        # ✅ close full writer first (safe even if None)
        if self._full_writer is not None:
            try:
                self._full_writer.release()
                self.logger.log(f"[FULL SAVED] {self.full_output_path}")
            except Exception as e:
                self.logger.log(f"[FULL ERROR] releasing full writer: {e}")
            self._full_writer = None

        # keep your existing 120s-clip save exactly as-is
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
