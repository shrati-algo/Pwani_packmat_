import cv2
import time
import logging
import numpy as np
import hashlib
import urllib.parse

logger = logging.getLogger("CameraLoader")


def sanitize_rtsp(rtsp_url: str) -> str:
    """
    Make RTSP credentials safe WITHOUT double-encoding.
    If password is already encoded (contains %40 etc.), keep it correct.
    """
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
    """
    Minimal-change multi-camera RTSP reader:
      ✅ keeps your exact interface:
         - start(num_threads=3)
         - get_latest_frame(cam_id=1)
         - stop()

      ✅ fixes most grey/stuck cases by:
         - frame hash stuck detection
         - low variance (grey/blank) detection
         - reconnect on repeated bad frames
         - draining internal buffer (grab N then read)
         - best-effort low-buffer / low-latency settings
    """

    def __init__(self, rtsp_url, width=1280, height=720):
        self.width = int(width)
        self.height = int(height)

        # Convert input into camera config like old design
        if isinstance(rtsp_url, str):
            self.config = {1: {"rtsp": rtsp_url}}
        else:
            self.config = {
                int(cam["cameraid"]): {"rtsp": cam.get("rtsp", "")}
                for cam in rtsp_url.values()
            }

        self.caps = {}
        self.frames = {}
        self.running = False

        # Per-cam watchdog state
        self.state = {}

        # Tunables (safe defaults)
        self.max_stuck = 8
        self.max_lowvar = 10
        self.max_read_fail = 8

        self.lowvar_std_threshold = 8
        self.reconnect_cooldown_sec = 2

        # Drain buffer: grab these many frames quickly, return the last decoded
        self.drain_grabs = 8

        # Compatibility with your index.py active check
        self.frame_set = {}

    def _init_state(self, cam_id: int):
        self.state[cam_id] = {
            "stuck": 0,
            "lowvar": 0,
            "fail": 0,
            "last_hash": None,
            "last_std": None,
            "restarts": 0,
            "last_ok_ts": 0.0,
        }
        self.frame_set[cam_id] = False

    def _frame_hash(self, frame) -> str:
        small = cv2.resize(frame, (64, 36))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return hashlib.md5(gray.tobytes()).hexdigest()

    def _frame_std(self, frame) -> float:
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))

    def _open_cap(self, cam_id: int):
        rtsp = self.config[cam_id]["rtsp"]
        if not rtsp:
            raise RuntimeError(f"Empty RTSP for camera {cam_id}")

        rtsp = sanitize_rtsp(rtsp)

        # Best-effort: reduce latency inside OpenCV
        # (some builds honor these, some don’t)
        cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)

        # Ask OpenCV to keep buffers tiny
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open RTSP stream for camera {cam_id}: {rtsp}")

        self.caps[cam_id] = cap
        self.frames[cam_id] = None
        logger.info(f"[CameraLoader] Opened cam={cam_id}")

    def _restart_cam(self, cam_id: int, reason: str):
        st = self.state[cam_id]
        st["restarts"] += 1
        logger.warning(f"[CameraLoader] Restart cam={cam_id} reason={reason} restarts={st['restarts']}")

        try:
            cap = self.caps.get(cam_id)
            if cap is not None:
                cap.release()
        except Exception:
            pass

        self.caps.pop(cam_id, None)
        time.sleep(self.reconnect_cooldown_sec)

        # reset counters but keep restart count
        self.state[cam_id].update({"stuck": 0, "lowvar": 0, "fail": 0, "last_hash": None})

        try:
            self._open_cap(cam_id)
        except Exception as e:
            logger.error(f"[CameraLoader] Restart failed cam={cam_id}: {e}")

    def start(self, num_threads=3):  # keep signature
        for cam_id in self.config.keys():
            if cam_id not in self.state:
                self._init_state(cam_id)

            rtsp = self.config[cam_id].get("rtsp", "")
            if not rtsp:
                logger.warning(f"[CameraLoader] cam={cam_id} RTSP empty; skipping start")
                continue

            self._open_cap(cam_id)

        self.running = True
        return self

    def get_latest_frame(self, cam_id=1):
        if not self.running:
            return None

        cam_id = int(cam_id)
        if cam_id not in self.state:
            self._init_state(cam_id)

        # ensure opened
        if cam_id not in self.caps:
            try:
                self._open_cap(cam_id)
            except Exception as e:
                logger.error(f"[CameraLoader] cam={cam_id} open failed: {e}")
                return None

        cap = self.caps.get(cam_id)
        if cap is None:
            return None

        # Drain buffer: grab a few frames quickly (drops older buffered frames)
        try:
            for _ in range(self.drain_grabs):
                cap.grab()
        except Exception:
            pass

        ret, frame = cap.read()
        if not ret or frame is None:
            st = self.state[cam_id]
            st["fail"] += 1
            logger.warning(f"[CameraLoader] cam={cam_id} read failed fail={st['fail']}/{self.max_read_fail}")
            if st["fail"] >= self.max_read_fail:
                self._restart_cam(cam_id, reason="read_fail")
            return None

        # Watchdog: stuck + low variance
        st = self.state[cam_id]
        h = self._frame_hash(frame)
        std = self._frame_std(frame)
        st["last_std"] = std

        if st["last_hash"] == h:
            st["stuck"] += 1
        else:
            st["stuck"] = 0
            st["last_hash"] = h

        if std < self.lowvar_std_threshold:
            st["lowvar"] += 1
        else:
            st["lowvar"] = 0

        st["fail"] = 0

        if st["lowvar"] >= self.max_lowvar:
            self._restart_cam(cam_id, reason=f"lowvar std={std:.2f}")
            return None

        if st["stuck"] >= self.max_stuck:
            self._restart_cam(cam_id, reason=f"stuck repeats={st['stuck']}")
            return None

        # OK
        st["last_ok_ts"] = time.time()
        self.frame_set[cam_id] = True
        self.frames[cam_id] = frame
        return frame

    def stop(self):
        self.running = False
        for cap in self.caps.values():
            try:
                cap.release()
            except Exception:
                pass
        self.caps.clear()
