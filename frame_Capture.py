import cv2

class CameraLoader:
    """
    Simple multi-camera RTSP reader without threads.
    Maintains only latest frame per camera.
    """
    def __init__(self, rtsp_url, width=1280, height=720):
        self.width = width
        self.height = height

        # Convert input into camera config like old design
        if isinstance(rtsp_url, str):
            self.config = {1: {"rtsp": rtsp_url}}
        else:
            self.config = {
                cam["cameraid"]: {"rtsp": cam["rtsp"]}
                for cam in rtsp_url.values()
            }

        self.caps = {}
        self.frames = {}
        self.running = False

    def start(self, num_threads=3):   # <--- KEEP old signature
        """Initialize VideoCapture for each camera (no threads)."""

        for cam_id, cfg in self.config.items():
            cap = cv2.VideoCapture(cfg["rtsp"], cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            if not cap.isOpened():
                raise RuntimeError(f"Could not open RTSP stream for camera {cam_id}")

            self.caps[cam_id] = cap
            self.frames[cam_id] = None

        self.running = True
        return self

    def get_latest_frame(self, cam_id=1):
        """Reads 1 frame on demand (no threads)."""
        if not self.running:
            return None

        cap = self.caps.get(cam_id)
        if cap is None:
            return None

        ret, frame = cap.read()
        if not ret:
            return None

        self.frames[cam_id] = frame
        return frame

    def stop(self):
        self.running = False
        for cap in self.caps.values():
            cap.release()
 
