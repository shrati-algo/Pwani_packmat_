import cv2
import numpy as np
import os
from ultralytics import YOLO
from datetime import datetime


# IOU calculation
def iou(b1, b2):
    x1, y1, x2, y2 = b1
    x1_p, y1_p, x2_p, y2_p = b2

    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    b1_area = (x2 - x1) * (y2 - y1)
    b2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = b1_area + b2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

# NMS
def apply_nms(detections, iou_thresh=0.5):
    filtered = []
    detections.sort(key=lambda x: x[2], reverse=True)
    while detections:
        best = detections.pop(0)
        filtered.append(best)
        detections = [d for d in detections if d[1] != best[1] or iou(d[0], best[0]) < iou_thresh]
    return filtered

# Tracker
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
                if current_iou > best_iou and current_iou > self.iou_threshold and obj_id not in used_ids:
                    best_iou = current_iou
                    best_id = obj_id

            cy = (bbox[1] + bbox[3]) // 2  # object center Y

            print(f"[TRACKING] Object center Y: {cy}, Line Y: {line_y}")

            if best_id is not None:
                updated_tracks[best_id] = {
                    'bbox': bbox,
                    'label': label,
                    'conf': conf,
                    'last_y': cy,
                    'missed': 0
                }

                if best_id not in self.counted_ids:
                    last_y = self.tracks[best_id]['last_y']
                    if last_y < line_y and cy >= line_y:
                        counter += 1
                        self.counted_ids.add(best_id)

                used_ids.add(best_id)
            else:
                updated_tracks[self.next_id] = {
                    'bbox': bbox,
                    'label': label,
                    'conf': conf,
                    'last_y': cy,
                    'missed': 0
                }
                self.next_id += 1

        for obj_id, data in self.tracks.items():
            if obj_id not in used_ids:
                data['missed'] += 1
                if data['missed'] < self.max_missed:
                    updated_tracks[obj_id] = data

        self.tracks = updated_tracks
        return counter

# Video Processor
class VideoProcessor:
    def __init__(self, video_path, model_path=r"packmat_i2.pt", camera_id=0):
        self.cap = cv2.VideoCapture(video_path)
        self.model = YOLO(model_path)
        self.camera_id = camera_id

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30

        # Set line dynamically at 75% of frame height
        self.line_y = int(self.frame_height * 0.75)
        self.line_start = (0, self.line_y)
        self.line_end = (self.frame_width, self.line_y)

        print(f"[INFO] Frame size: {self.frame_width}x{self.frame_height}, Line Y: {self.line_y}")

        self.counter = 0
        self.tracker = ObjectTracker()

        # Output video writer
        os.makedirs("outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = f"cam_{self.camera_id}_{timestamp}_output.mp4"
        self.output_path = os.path.join("outputs", output_filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.frame_width, self.frame_height))

    def process_video(self):
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open video stream.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Stream ended or interrupted.")
                break

            results = self.model(frame, conf=0.25)[0]
            detections = []

            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                conf = float(box.conf[0])
                if label.lower() in ["jerrycan_bundle", "carton", "carton_brown"] and conf > 0.6:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(((x1, y1, x2, y2), label, conf))

            detections = apply_nms(detections, iou_thresh=0.5)

            # Draw counting line
            cv2.line(frame, self.line_start, self.line_end, (0, 0, 255), 2)
            #cv2.putText(frame, "COUNTING LINE", (self.line_start[0] + 10, self.line_y - 10),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            self.counter = self.tracker.update_tracks(detections, self.line_y, self.counter)

            for obj_id, data in self.tracker.tracks.items():
                x1, y1, x2, y2 = data['bbox']
                label = data['label']
                conf = data['conf']
                color = (0, 255, 0) if label == "jerrycan_bundle" else (255, 255, 0)
                label_text = f"{label} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Smaller counter display
            cv2.putText(frame, f"Counter: {self.counter}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            self.out.write(frame)

        self.cleanup()
        return self.counter

    def cleanup(self):
        self.cap.release()
        self.out.release()


# # Standalone usage
# if __name__ == "__main__":
#     try:
#         rtsp_link = get_next_video()
#         if not rtsp_link:
#             raise ValueError("RTSP link not found.")

#         print(f"Processing RTSP stream from: {rtsp_link}")
#         processor = VideoProcessor(video_path=rtsp_link, camera_id=1)
#         count = processor.process_video()
#         print(f"Total objects counted: {count}")
#         print(f"Output video saved to: {processor.output_path}")

#     except Exception as e:
#         print(f"[ERROR] {str(e)}")

