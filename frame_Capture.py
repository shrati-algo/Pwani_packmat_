import subprocess
import threading
import queue
import urllib.parse
import logging
import numpy as np
import time

logger = logging.getLogger("CameraLoader")


def sanitize_rtsp(rtsp_url: str) -> str:
    """
    Encode special characters in RTSP password.
    Example:
      rtsp://user:PW@n@ip/...  ->  rtsp://user:PW%40n@ip/...
    """
    try:
        scheme, rest = rtsp_url.split("://", 1)
        creds, host = rest.split("@", 1)
        user, pwd = creds.split(":", 1)
        pwd_encoded = urllib.parse.quote(pwd, safe="")
        return f"{scheme}://{user}:{pwd_encoded}@{host}"
    except Exception:
        return rtsp_url


class CameraLoader:
    """
    FFmpeg-based RTSP reader:
      - Forces output resolution with -vf scale=W:H (fixes scrambled/static frames)
      - Produces writable numpy frames (.copy()) so OpenCV can draw on them
      - Keeps latency low by dropping old frames (only latest is kept)
      - Auto-restarts FFmpeg if stream glitches
    """

    def __init__(self, rtsp_url, width=1280, height=720, queue_size=2, reconnect_delay=0.5):
        self.rtsp_url = sanitize_rtsp(rtsp_url)
        self.width = int(width)
        self.height = int(height)

        self.frame_queue = queue.Queue(maxsize=int(queue_size))
        self.running = False
        self.thread = None
        self.process = None

        self.reconnect_delay = float(reconnect_delay)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        logger.info("CameraLoader started")
        return self

    def _start_ffmpeg(self):
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-loglevel", "error",
            "-an",
            "-i", self.rtsp_url,

            # ✅ CRITICAL: force a fixed output size so numpy reshape matches
            "-vf", f"scale={self.width}:{self.height}",

            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-"
        ]
        logger.info("Starting FFmpeg stream (scaled to %dx%d)", self.width, self.height)
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )

    def _reader(self):
        frame_size = self.width * self.height * 3

        while self.running:
            # (re)start ffmpeg if needed
            if self.process is None or self.process.poll() is not None:
                try:
                    if self.process:
                        self.process.kill()
                except Exception:
                    pass

                self.process = self._start_ffmpeg()
                time.sleep(0.2)

            raw_frame = self.process.stdout.read(frame_size)

            # If we didn't get a full frame, restart ffmpeg
            if len(raw_frame) != frame_size:
                try:
                    self.process.kill()
                except Exception:
                    pass
                self.process = None
                time.sleep(self.reconnect_delay)
                continue

            # ✅ writable frame for cv2.line/rectangle/putText
            frame = np.frombuffer(raw_frame, np.uint8).reshape((self.height, self.width, 3)).copy()

            # Keep only the latest frame to avoid lag
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break

            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

        logger.info("Camera thread stopped")

    def get_latest_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        try:
            if self.process:
                self.process.kill()
        except Exception:
            pass
        logger.info("CameraLoader stopped")
