import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from datetime import datetime
from gStreamer import get_gst_pipeline

# -------------------------------
# IOU Calculation
# -------------------------------
def iou(b1, b2):
    x1, y1, x2, y2 = b1
    x1p, y1p, x2p, y2p = b2
    xi1, yi1 = max(x1, x1p), max(y1, y1p)
    xi2, yi2 = min(x2, x2p), min(y2, y2p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    b1_area = (x2 - x1) * (y2 - y1)
    b2_area = (x2p - x1p) * (y2p - y1p)
    union_area = b1_area + b2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# -------------------------------
# NMS
# -------------------------------
def apply_nms(detections, iou_thresh=0.5):
    filtered = []
    detections.sort(key=lambda x: x[2], reverse=True)
    while detections:
        best = detections.pop(0)
        filtered.append(best)
        detections = [
            d for d in detections
            if d[1] != best[1] or iou(d[0], best[0]) < iou_thresh
        ]
    return filtered

# -------------------------------
# Simple Tracker
# -------------------------------
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
                current_iou = iou(bbox, data['bbox'])
                if (current_iou > best_iou and current_iou > self.iou_threshold and obj_id not in used_ids):
                    best_iou = current_iou
                    best_id = obj_id

            cy = (bbox[1] + bbox[3]) // 2

            if best_id is not None:
                updated_tracks[best_id] = {
                    'bbox': bbox, 'label': label, 'conf': conf,
                    'last_y': cy, 'missed': 0
                }
                if best_id not in self.counted_ids:
                    last_y = self.tracks[best_id]['last_y']
                    if last_y < line_y and cy >= line_y:
                        counter += 1
                        self.counted_ids.add(best_id)
                        print(f"[COUNTED] ID {best_id} crossed. Count={counter}")
                used_ids.add(best_id)
            else:
                updated_tracks[self.next_id] = {
                    'bbox': bbox, 'label': label, 'conf': conf,
                    'last_y': cy, 'missed': 0
                }
                self.next_id += 1

        for obj_id, data in self.tracks.items():
            if obj_id not in used_ids:
                data['missed'] += 1
                if data['missed'] < self.max_missed:
                    updated_tracks[obj_id] = data

        self.tracks = updated_tracks
        return counter

# -------------------------------
# Main Processor
# -------------------------------
class VideoProcessor:
    def __init__(self, rtsp_url, model_path="packmat_i2.pt", camera_id=0):
        self.gst_pipeline = get_gst_pipeline(
            rtsp_url=rtsp_url, drop_frames=True, latency=0
        )
        self.cap = cv2.VideoCapture(self.gst_pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("[ERROR] Could not open RTSP stream")

        self.model = YOLO(model_path).to("cuda")
        self.camera_id = camera_id
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30

        self.line_y = int(self.frame_height * 0.75)
        self.line_start = (0, self.line_y)
        self.line_end = (self.frame_width, self.line_y)

        print(f"[INFO] Camera {camera_id} - {self.frame_width}x{self.frame_height} @ {self.fps}fps")

        self.counter = 0
        self.tracker = ObjectTracker()

        os.makedirs("outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"cam_{camera_id}_{timestamp}_annotated.avi"
        self.output_path = os.path.join("outputs", output_filename)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps,
                                   (self.frame_width, self.frame_height))

        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def _reconnect(self):
        print(f"[WARN] Reconnecting to camera {self.camera_id}...")
        self.cap.release()
        time.sleep(1)
        self.cap = cv2.VideoCapture(self.gst_pipeline, cv2.CAP_GSTREAMER)
        if self.cap.isOpened():
            print(f"[INFO] Reconnected to camera {self.camera_id}")
        else:
            print(f"[ERROR] Failed to reconnect to camera {self.camera_id}")

    def process_video(self):
        frame_skip = 2
        frame_count = 0

        while not self._stop_flag:
            ret, frame = self.cap.read()
            if not ret:
                self._reconnect()
                continue

            detections = []

            if frame_count % frame_skip == 0:
                orig_h, orig_w = frame.shape[:2]
                resized_frame = cv2.resize(frame, (640, 640))

                # Debug timing start
                start_time = time.time()
                results = self.model(resized_frame, conf=0.25, imgsz=640, device=0)[0]
                inf_time_ms = (time.time() - start_time) * 1000
                print(f"[{self.camera_id}] Inference time: {inf_time_ms:.2f} ms, "
                      f"Detections: {len(results.boxes)}")

                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id]
                    conf = float(box.conf[0])

                    if label.lower() in ["jerrycan_bundle", "carton", "carton_brown"] and conf > 0.6:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        scale_x, scale_y = orig_w / 640, orig_h / 640
                        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                        detections.append(((x1, y1, x2, y2), label, conf))

                detections = apply_nms(detections, iou_thresh=0.5)

            # Draw counting line
            cv2.line(frame, self.line_start, self.line_end, (0, 0, 255), 2)

            # Update tracks & counter
            self.counter = self.tracker.update_tracks(detections, self.line_y, self.counter)

            # Draw tracked objects
            for obj_id, data in self.tracker.tracks.items():
                x1, y1, x2, y2 = data['bbox']
                label, conf = data['label'], data['conf']
                color = (0, 255, 0) if label == "jerrycan_bundle" else (255, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.putText(frame, f"Counter: {self.counter}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            self.out.write(frame)
            frame_count += 1

        self.cleanup()
        return self.counter

    def cleanup(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
