import cv2
from collections import deque
from threading import Thread
import time


class CameraLoader:
    """
    RTSP Camera Loader (Crash-safe, Low-latency)
    """

    def __init__(self, rtsp_url, camera_id=1):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id

        self.video_objects = {}
        self.frame_set = {camera_id: deque(maxlen=1)}
        self.stopped = False

        self._open_camera()

    # -------------------------------------------------
    def _open_camera(self):
        """Open RTSP stream safely"""
        print("[INFO] Opening RTSP stream...")

        cap = cv2.VideoCapture(
            self.rtsp_url,
            cv2.CAP_FFMPEG
        )

        # 🔴 CRITICAL FIXES
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))

        if not cap.isOpened():
            raise RuntimeError("[ERROR] Failed to open RTSP stream")

        self.video_objects[self.camera_id] = {
            "capture_obj": cap,
            "cameraID": self.camera_id
        }

        print("[INFO] RTSP stream opened successfully")

    # -------------------------------------------------
    def start(self):
        Thread(target=self._update_frames, daemon=True).start()
        return self

    # -------------------------------------------------
    def _update_frames(self):
        while not self.stopped:
            cap_obj = self.video_objects.get(self.camera_id)
            cap = cap_obj["capture_obj"]

            try:
                ret, frame = cap.read()

                if not ret:
                    print("[WARN] Frame grab failed, reconnecting RTSP...")
                    self._reconnect()
                    continue

                self.frame_set[self.camera_id].append(frame)

            except Exception as e:
                print(f"[ERROR] RTSP read crash: {e}")
                self._reconnect()

            time.sleep(0.001)

    # -------------------------------------------------
    def _reconnect(self):
        """Reconnect RTSP safely"""
        try:
            cap = self.video_objects[self.camera_id]["capture_obj"]
            cap.release()
        except:
            pass

        time.sleep(1)
        self._open_camera()

    # -------------------------------------------------
    def get_latest_frame(self, camera_id=1):
        if camera_id in self.frame_set and len(self.frame_set[camera_id]) > 0:
            return self.frame_set[camera_id][-1]
        return None

    # -------------------------------------------------
    def stop(self):
        self.stopped = True

        try:
            self.video_objects[self.camera_id]["capture_obj"].release()
        except:
            pass

        print("[INFO] CameraLoader stopped")
