import subprocess
import threading
import queue
import urllib.parse
import logging
import time
from collections import deque
import numpy as np

logger = logging.getLogger("CameraLoader")


def sanitize_rtsp(rtsp_url: str) -> str:
    """
    Make RTSP credentials safe WITHOUT double-encoding.
    If password is already encoded (contains %40 etc.), this keeps it correct.
    """
    try:
        scheme, rest = rtsp_url.split("://", 1)
        creds, host = rest.split("@", 1)
        user, pwd = creds.split(":", 1)

        # ✅ decode first to avoid double-encoding, then encode safely
        pwd_decoded = urllib.parse.unquote(pwd)
        pwd_encoded = urllib.parse.quote(pwd_decoded, safe="")

        return f"{scheme}://{user}:{pwd_encoded}@{host}"
    except Exception:
        return rtsp_url



class CameraLoader:
    """
    FFmpeg-based RTSP reader:
      - Forces output resolution with -vf scale=W:H
      - Produces writable numpy frames (.copy())
      - Keeps latency low (only latest frame)
      - Auto-restarts FFmpeg if stream glitches
      - Supports SINGLE or MULTI camera transparently
    """

    def __init__(self, rtsp_url, width=1280, height=720, queue_size=2, reconnect_delay=0.5):
        self.width = int(width)
        self.height = int(height)
        self.reconnect_delay = float(reconnect_delay)
        self.queue_size = int(queue_size)

        self.running = False
        self.lock = threading.Lock()

        # ---- Normalize input to multi-camera config ----
        if isinstance(rtsp_url, str):
            self.config = {
                1: {"rtsp": sanitize_rtsp(rtsp_url)}
            }
        elif isinstance(rtsp_url, dict):
            self.config = {
                cfg["cameraid"]: {"rtsp": sanitize_rtsp(cfg["rtsp"])}
                for cfg in rtsp_url.values()
            }
        else:
            raise ValueError("rtsp_url must be str or dict")

        # Per-camera state
        self.processes = {}
        self.frame_set = {cid: deque(maxlen=1) for cid in self.config}
        self.retry_count = {cid: 0 for cid in self.config}
        self.last_ok = {cid: 0 for cid in self.config}
        self.disabled = {cid: False for cid in self.config}

        # Thread pool
        self.task_queue = queue.Queue()
        self.workers = []

    # --------------------------------------------------
    def _start_ffmpeg(self, rtsp):
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-loglevel", "error",
            "-an",
            "-i", rtsp,
            "-vf", f"scale={self.width}:{self.height}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-"
        ]
        try:
            return subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**8
            )
        except Exception:
            return None

    def _kill_ffmpeg(self, proc):
        try:
            if proc and proc.poll() is None:
                proc.kill()
        except Exception:
            pass

    # --------------------------------------------------
    def start(self, num_threads=3):
        self.running = True

        # Start ffmpeg processes
        for cam_id, cfg in self.config.items():
            proc = self._start_ffmpeg(cfg["rtsp"])
            self.processes[cam_id] = proc

        # Start workers
        for i in range(num_threads):
            t = threading.Thread(target=self._worker, args=(i,), daemon=True)
            t.start()
            self.workers.append(t)

        for cam_id in self.config:
            self.task_queue.put(cam_id)

        logger.info("CameraLoader started with %d cameras", len(self.config))
        return self

    # --------------------------------------------------
    def _worker(self, wid):
        frame_size = self.width * self.height * 3

        while self.running:
            try:
                cam_id = self.task_queue.get(timeout=1)
            except queue.Empty:
                continue

            if self.disabled[cam_id]:
                time.sleep(2)
                self.task_queue.put(cam_id)
                continue

            proc = self.processes.get(cam_id)

            if proc is None or proc.poll() is not None:
                self._restart(cam_id)
                self.task_queue.put(cam_id)
                continue

            try:
                raw = proc.stdout.read(frame_size)
            except Exception:
                self._restart(cam_id)
                self.task_queue.put(cam_id)
                continue

            if not raw or len(raw) != frame_size:
                if time.time() - self.last_ok[cam_id] > 3:
                    self._restart(cam_id)
                self.task_queue.put(cam_id)
                continue

            frame = np.frombuffer(raw, np.uint8)\
                      .reshape((self.height, self.width, 3))\
                      .copy()

            with self.lock:
                self.frame_set[cam_id].clear()
                self.frame_set[cam_id].append(frame)
                self.retry_count[cam_id] = 0
                self.last_ok[cam_id] = time.time()

            self.task_queue.put(cam_id)

    # --------------------------------------------------
    def _restart(self, cam_id):
        self.retry_count[cam_id] += 1
        self._kill_ffmpeg(self.processes.get(cam_id))

        if self.retry_count[cam_id] > 3:
            logger.error("Camera %d permanently disabled", cam_id)
            self.disabled[cam_id] = True
            return

        time.sleep(self.reconnect_delay)
        self.processes[cam_id] = self._start_ffmpeg(self.config[cam_id]["rtsp"])
        logger.warning("Restarting camera %d (%d)", cam_id, self.retry_count[cam_id])

    # --------------------------------------------------
    def get_latest_frame(self, cam_id=1):
        with self.lock:
            frames = self.frame_set.get(cam_id)
            if frames:
                return frames[0]
        return None

    # --------------------------------------------------
    def stop(self):
        self.running = False
        for p in self.processes.values():
            self._kill_ffmpeg(p)
        logger.info("CameraLoader stopped")
