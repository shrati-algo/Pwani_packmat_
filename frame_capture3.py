import subprocess
import numpy as np
import time
import threading
import queue
import logging
import urllib.parse

logger = logging.getLogger("CameraPipeline")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)


# --------------------------------------------------
# RTSP Sanitizer
# --------------------------------------------------
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


# --------------------------------------------------
# GPU Camera Worker (FFmpeg + NVDEC)
# --------------------------------------------------
class GPUCameraWorker(threading.Thread):
    def __init__(self, cam_id, rtsp, frame_queue, width=1280, height=720, codec="h264"):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.rtsp = sanitize_rtsp(rtsp)
        self.queue = frame_queue
        self.width = width
        self.height = height
        self.running = True
        self.process = None
        self.frame_size = self.width * self.height * 3
        self.last_frame_time = time.time()
        self.restart_count = 0
        self.no_frame_timeout = 5
        self.codec = codec.lower()

    # --------------------------------------------------
    def start_ffmpeg(self):
        logger.info(f"[Cam {self.cam_id}] Starting GPU FFmpeg ({self.codec})...")

        hw_decoder = "h264_cuvid" if self.codec == "h264" else "hevc_cuvid"

        command = [
            "ffmpeg",
            "-loglevel", "error",
            "-hwaccel", "cuda",
            "-c:v", hw_decoder,
            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-i", self.rtsp,
            "-vf", f"scale_cuda={self.width}:{self.height}",
            "-pix_fmt", "bgr24",
            "-vsync", "0",
            "-f", "rawvideo",
            "pipe:1"
        ]

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )

    # --------------------------------------------------
    def restart(self, reason):
        self.restart_count += 1
        logger.warning(f"[Cam {self.cam_id}] RESTART | reason={reason} | count={self.restart_count}")
        try:
            if self.process:
                self.process.kill()
        except Exception:
            pass
        self.process = None
        time.sleep(2)

    # --------------------------------------------------
    def run(self):
        while self.running:
            try:
                if self.process is None:
                    self.start_ffmpeg()

                raw = self.process.stdout.read(self.frame_size)
                if len(raw) != self.frame_size:
                    raise RuntimeError("Incomplete frame")

                frame = np.frombuffer(raw, np.uint8).reshape((self.height, self.width, 3))
                self.last_frame_time = time.time()

                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except:
                        pass

                self.queue.put(frame)

            except Exception as e:
                logger.error(f"[Cam {self.cam_id}] ERROR: {e}")
                self.restart("decode_fail")

    # --------------------------------------------------
    def is_alive_stream(self):
        return (time.time() - self.last_frame_time) < self.no_frame_timeout

    # --------------------------------------------------
    def stop(self):
        self.running = False
        try:
            if self.process:
                self.process.kill()
        except:
            pass


# --------------------------------------------------
# Camera Pipeline (Supervisor)
# --------------------------------------------------
class CameraPipeline:
    def __init__(self, rtsp_dict, queue_size=5, default_width=1280, default_height=720):
        """
        rtsp_dict format:
        {
            "cam1": {"rtsp": "...", "cameraid": 1, "codec":"h264"},
            "cam2": {"rtsp": "...", "cameraid": 2, "codec":"h265"}
        }
        """
        self.rtsp_dict = rtsp_dict
        self.queue_size = queue_size
        self.default_width = default_width
        self.default_height = default_height

        self.workers = {}
        self.queues = {}

        self.running = False
        self.monitor_thread = None

    # --------------------------------------------------
    def start(self):
        logger.info("[Pipeline] Starting GPU pipeline...")
        self.running = True

        for name, cfg in self.rtsp_dict.items():
            cam_id = int(cfg.get("cameraid", 0))
            rtsp = cfg.get("rtsp", "")
            codec = cfg.get("codec", "h264")

            if not rtsp:
                logger.warning(f"[Pipeline] cam={cam_id} RTSP empty, skipping")
                continue

            q = queue.Queue(maxsize=self.queue_size)
            worker = GPUCameraWorker(cam_id, rtsp, q, self.default_width, self.default_height, codec)
            worker.start()

            self.workers[cam_id] = worker
            self.queues[cam_id] = q

        # start monitor
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("[Pipeline] GPU pipeline started")

    # --------------------------------------------------
    def _monitor_loop(self):
        while self.running:
            try:
                for cam_id, worker in self.workers.items():
                    if not worker.is_alive_stream():
                        logger.warning(f"[Monitor] cam={cam_id} no frames → restarting")
                        worker.restart("watchdog_no_frame")
                time.sleep(2)
            except Exception as e:
                logger.error(f"[Monitor] ERROR: {e}")

    # --------------------------------------------------
    def get_frame(self, cam_id, timeout=1.0):
        q = self.queues.get(int(cam_id))
        if not q:
            return None
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            logger.warning(f"[Pipeline] cam={cam_id} no frame available")
            return None

    # --------------------------------------------------
    def is_camera_alive(self, cam_id):
        worker = self.workers.get(int(cam_id))
        if not worker:
            return False
        return worker.is_alive_stream()

    # --------------------------------------------------
    def stop(self):
        logger.info("[Pipeline] Stopping pipeline...")
        self.running = False
        for worker in self.workers.values():
            worker.stop()
        logger.info("[Pipeline] Stopped")