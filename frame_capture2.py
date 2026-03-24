import cv2
import time
import logging
import numpy as np
import hashlib
import urllib.parse
import threading

logger = logging.getLogger("CameraLoader")


def sanitize_rtsp(rtsp_url: str) -> str:
    try:
        scheme, rest = rtsp_url.split("://", 1)
        creds, host = rest.split("@", 1)
        user, pwd = creds.split(":", 1)

        pwd_decoded = urllib.parse.unquote(pwd)
        pwd_encoded = urllib.parse.quote(pwd_decoded, safe="")

        return f"{scheme}://{user}:{pwd_encoded}@{host}"
    except Exception:
        return rtsp_url


class CameraLoader:
    def __init__(self, rtsp_url, width=1280, height=720):
        self.width = int(width)
        self.height = int(height)

        if isinstance(rtsp_url, str):
            self.config = {1: {"rtsp": rtsp_url}}
        else:
            self.config = {
                int(cam["cameraid"]): {"rtsp": cam.get("rtsp", "")}
                for cam in rtsp_url.values()
            }

        self.frames = {}
        self.running = False
        self.threads = {}
        self.locks = {}

        self.state = {}
        self.frame_set = {}

        # Tunables
        self.read_timeout = 2
        self.reconnect_delay = 2

        self.max_stuck = 6
        self.max_lowvar = 8
        self.lowvar_threshold = 8

    # ================= STATE =================

    def _init_state(self, cam_id):
        self.state[cam_id] = {
            "stuck": 0,
            "lowvar": 0,
            "last_hash": None,
            "last_ok_ts": 0,
            "restarts": 0
        }
        self.frame_set[cam_id] = False

    # ================= METRICS =================

    def _frame_hash(self, frame):
        small = cv2.resize(frame, (64, 36))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return hashlib.md5(gray.tobytes()).hexdigest()

    def _frame_std(self, frame):
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))

    # ================= SAFE READ =================

    def _safe_read(self, cap):
        result = {"ret": False, "frame": None}

        def target():
            try:
                result["ret"], result["frame"] = cap.read()
            except Exception:
                result["ret"], result["frame"] = False, None

        t = threading.Thread(target=target)
        t.daemon = True
        t.start()
        t.join(self.read_timeout)

        if t.is_alive():
            return False, None, True

        return result["ret"], result["frame"], False

    # ================= CAMERA THREAD =================

    def _camera_worker(self, cam_id):
        rtsp = sanitize_rtsp(self.config[cam_id]["rtsp"])

        while self.running:
            try:
                logger.info(f"[CameraLoader] Starting cam={cam_id}")
                cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)

                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

                if not cap.isOpened():
                    raise RuntimeError("Failed to open stream")

                st = self.state[cam_id]

                while self.running:
                    ret, frame, timeout = self._safe_read(cap)

                    if timeout:
                        logger.error(f"[CameraLoader] cam={cam_id} TIMEOUT")
                        break

                    if not ret or frame is None:
                        logger.warning(f"[CameraLoader] cam={cam_id} read failed")
                        break

                    # Frame checks
                    h = self._frame_hash(frame)
                    std = self._frame_std(frame)

                    if st["last_hash"] == h:
                        st["stuck"] += 1
                    else:
                        st["stuck"] = 0
                        st["last_hash"] = h

                    if std < self.lowvar_threshold:
                        st["lowvar"] += 1
                    else:
                        st["lowvar"] = 0

                    if st["stuck"] >= self.max_stuck:
                        logger.warning(f"[CameraLoader] cam={cam_id} stuck frames")
                        break

                    if st["lowvar"] >= self.max_lowvar:
                        logger.warning(f"[CameraLoader] cam={cam_id} low variance")
                        break

                    st["last_ok_ts"] = time.time()

                    # Save latest frame (thread-safe)
                    with self.locks[cam_id]:
                        self.frames[cam_id] = frame
                        self.frame_set[cam_id] = True

                    time.sleep(0.01)

                cap.release()

            except Exception as e:
                logger.error(f"[CameraLoader] cam={cam_id} error: {e}")

            # reconnect
            self.state[cam_id]["restarts"] += 1
            logger.info(f"[CameraLoader] Reconnecting cam={cam_id} attempt={self.state[cam_id]['restarts']}")

            time.sleep(self.reconnect_delay)

    # ================= PUBLIC =================

    def start(self, num_threads=3):
        logger.info("[CameraLoader] Starting threaded cameras")
        self.running = True

        for cam_id in self.config.keys():
            self._init_state(cam_id)
            self.frames[cam_id] = None
            self.locks[cam_id] = threading.Lock()

            t = threading.Thread(target=self._camera_worker, args=(cam_id,))
            t.daemon = True
            t.start()

            self.threads[cam_id] = t

        return self

    def get_latest_frame(self, cam_id=1):
        cam_id = int(cam_id)

        if cam_id not in self.frames:
            return None

        with self.locks[cam_id]:
            return self.frames[cam_id]

    def stop(self):
        logger.info("[CameraLoader] Stopping all cameras")
        self.running = False

        for t in self.threads.values():
            t.join(timeout=1)

        self.threads.clear()