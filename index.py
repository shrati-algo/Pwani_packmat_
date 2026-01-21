#!/usr/bin/env python3

import os
import time
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from get_rtsp_link import get_rtsp_link
from save_to_DB import save_video_log
from video_tracker import mark_video_as_processed

# Your parallel multi-camera ffmpeg loader (dict input)
from frame_Capture import CameraLoader

# Your existing processor (unchanged)
from packmat_counter import VideoProcessor


# --------------------------------------------------
# Logging (file + console)
# --------------------------------------------------
LOG_FILE = os.environ.get("SERVICE_LOG", "service.log")

logger = logging.getLogger("packmat_service")
logger.setLevel(logging.INFO)

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

if not logger.handlers:
    _file = logging.FileHandler(LOG_FILE)
    _file.setFormatter(_fmt)
    logger.addHandler(_file)

    _console = logging.StreamHandler()
    _console.setFormatter(_fmt)
    logger.addHandler(_console)


# --------------------------------------------------
# FastAPI
# --------------------------------------------------
app = FastAPI(title="Packmat Service (Parallel 5 Cameras)")


# --------------------------------------------------
# Tunables (adjust if needed)
# --------------------------------------------------
FIXED_CAMERA_IDS = [1, 2, 3, 4, 5]   # fixed cameras

CAMERA_LOADER_THREADS = 3           # decoding workers inside CameraLoader

DEFAULT_FPS = 20
DEFAULT_FRAME_SKIP = 2

JOB_LOOP_SLEEP = 0.001              # per-job loop pacing
STATUS_UPDATE_EVERY_N_LOOPS = 10    # reduce lock contention

NO_FRAME_WARN_SECS = 5              # warn if no frames for long


# --------------------------------------------------
# Job model
# --------------------------------------------------
@dataclass
class JobState:
    camera_id: str
    truck_visit_id: Optional[str] = None

    status: str = "idle"  # idle | starting | running | stopping | completed | error
    count: int = 0
    output_path: Optional[str] = None
    video_link: Optional[str] = None
    started_at: Optional[str] = None
    last_log: Optional[str] = None
    error: Optional[str] = None

    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None

    # per-job processor (IMPORTANT: separate for each cam)
    processor: Optional[VideoProcessor] = None


def job_public(job: JobState) -> Dict[str, Any]:
    return {
        "camera_id": job.camera_id,
        "truck_visit_id": job.truck_visit_id,
        "status": job.status,
        "count": job.count,
        "output_path": job.output_path,
        "video_link": job.video_link,
        "started_at": job.started_at,
        "last_log": job.last_log,
        "error": job.error,
    }


# --------------------------------------------------
# Global shared state
# --------------------------------------------------
jobs_lock = threading.Lock()
jobs: Dict[str, JobState] = {}  # key=camera_id ("1".."5")

camera_loader: Optional[CameraLoader] = None
camera_loader_lock = threading.Lock()

RTSP_DICT: Dict[str, Dict[str, Any]] = {}


# --------------------------------------------------
# Build fixed RTSP dict once
# --------------------------------------------------
def build_fixed_rtsp_dict() -> Dict[str, Dict[str, Any]]:
    cams: Dict[str, Dict[str, Any]] = {}

    for cid in FIXED_CAMERA_IDS:
        rtsp = get_rtsp_link(str(cid))

        if not rtsp:
            logger.warning("[INIT] RTSP not found for camera_id=%s (camera may stay inactive)", cid)
            rtsp = ""  # keep entry so we know camera exists

        cams[f"cam{cid}"] = {"rtsp": rtsp, "cameraid": cid}

    return cams


# --------------------------------------------------
# Ensure shared camera loader running (one-time)
# --------------------------------------------------
def ensure_camera_loader_running():
    global camera_loader, RTSP_DICT

    with camera_loader_lock:
        if camera_loader is not None:
            return

        RTSP_DICT = build_fixed_rtsp_dict()
        logger.info("[INIT] Starting shared multi-camera CameraLoader for %s cameras...", len(RTSP_DICT))

        for name, cfg in RTSP_DICT.items():
            logger.info("[INIT] %s -> cameraid=%s rtsp=%s", name, cfg.get("cameraid"), "SET" if cfg.get("rtsp") else "EMPTY")

        camera_loader = CameraLoader(RTSP_DICT)
        camera_loader.start(num_threads=CAMERA_LOADER_THREADS)

        time.sleep(1.0)

        active = []
        try:
            for cid, frames in camera_loader.frame_set.items():
                if frames:
                    active.append(cid)
        except Exception:
            pass

        logger.info("[INIT] Shared CameraLoader started. Active cams (initial frames): %s", active)


# --------------------------------------------------
# Per-camera job worker
# --------------------------------------------------
def camera_job_worker(job: JobState):
    cam_id = job.camera_id
    tv_id = job.truck_visit_id

    logger.info("[%s] JOB START tv_id=%s", cam_id, tv_id)

    try:
        ensure_camera_loader_running()

        # Each camera job owns its own processor/counter/buffer
        processor = VideoProcessor(
            model_path="packmat_i2.pt",
            camera_id=cam_id,
            fps=DEFAULT_FPS,
            frame_skip=DEFAULT_FRAME_SKIP
        )
        job.processor = processor

        with jobs_lock:
            job.status = "running"
            job.started_at = datetime.utcnow().isoformat()
            job.last_log = "running"
            job.error = None

        last_frame_ts = time.time()
        frames_seen = 0
        loops = 0
        last_print = time.time()

        while not job.stop_event.is_set():
            loops += 1

            # Pull latest frame for THIS camera from shared loader
            try:
                frame = camera_loader.get_latest_frame(int(cam_id))
            except Exception:
                frame = None

            if frame is None:
                if time.time() - last_frame_ts > NO_FRAME_WARN_SECS:
                    logger.warning("[%s] No frames received for >%ss", cam_id, NO_FRAME_WARN_SECS)
                    last_frame_ts = time.time()
                    with jobs_lock:
                        job.last_log = f"no frames >{NO_FRAME_WARN_SECS}s"
                time.sleep(0.01)
                continue

            frames_seen += 1
            last_frame_ts = time.time()

            # Process frame (your logic)
            try:
                count, output_path = processor.process_frame(frame)
            except Exception as e:
                logger.exception("[%s] process_frame crashed", cam_id)
                with jobs_lock:
                    job.status = "error"
                    job.error = "process_frame crashed"
                    job.last_log = str(e)[:250]
                break

            # Video link
            if output_path:
                video_name = os.path.basename(output_path)
                video_link = f"http://192.168.5.82:5009/{video_name}"
            else:
                video_link = None

            # Status update throttled
            if loops % STATUS_UPDATE_EVERY_N_LOOPS == 0:
                with jobs_lock:
                    job.count = int(count) if count is not None else job.count
                    job.output_path = output_path
                    job.video_link = video_link
                    job.last_log = f"frames_seen={frames_seen}"

            # Light periodic prints (so you see it’s alive)
            if time.time() - last_print > 5:
                logger.info("[%s] Running | frames_seen=%s | count=%s", cam_id, frames_seen, job.count)
                last_print = time.time()

            time.sleep(JOB_LOOP_SLEEP)

        # Stop requested (or error)
        logger.info("[%s] Stop requested. Finalizing last 60s video...", cam_id)
        with jobs_lock:
            if job.status != "error":
                job.status = "stopping"
            job.last_log = "finalizing video"

        # Finalize
        try:
            processor.cleanup()
        except Exception:
            logger.exception("[%s] processor.cleanup failed", cam_id)
            with jobs_lock:
                job.status = "error"
                job.error = "processor.cleanup failed"

        # Snapshot + DB log
        with jobs_lock:
            if job.status != "error":
                job.status = "completed"
                job.last_log = "completed"

            final_count = job.count
            final_output = job.output_path
            final_link = job.video_link

        try:
            save_video_log(tv_id, final_output, final_count, final_link)
            logger.info("[%s] DB log saved tv_id=%s count=%s", cam_id, tv_id, final_count)
        except Exception:
            logger.exception("[%s] save_video_log failed", cam_id)
            with jobs_lock:
                job.last_log = "save_video_log failed"

        try:
            if final_output:
                mark_video_as_processed(final_output)
                logger.info("[%s] Marked processed: %s", cam_id, final_output)
        except Exception:
            logger.exception("[%s] mark_video_as_processed failed", cam_id)
            with jobs_lock:
                job.last_log = "mark_video_as_processed failed"

        logger.info("[%s] JOB END status=%s", cam_id, job.status)

    except Exception as e:
        logger.exception("[%s] JOB OUTER ERROR", cam_id)
        with jobs_lock:
            job.status = "error"
            job.error = "job outer exception"
            job.last_log = str(e)[:250]

    finally:
        logger.info("[%s] JOB EXIT", cam_id)


# --------------------------------------------------
# API: START (parallel per camera)
# --------------------------------------------------
@app.post("/process_packmat")
async def process_packmat(request: Request):
    data = await request.json()

    if not data or "trigger" not in data or "Conveyr_id" not in data or "truck_visit_id" not in data:
        raise HTTPException(status_code=400, detail="Missing required parameters")

    if data["trigger"] == 0:
        return JSONResponse({"status": "ignored", "message": "Trigger was 0"})

    camera_id = str(data["Conveyr_id"])
    truck_visit_id = str(data["truck_visit_id"])

    if camera_id not in {str(x) for x in FIXED_CAMERA_IDS}:
        raise HTTPException(status_code=400, detail=f"Invalid Conveyr_id={camera_id}. Allowed: {FIXED_CAMERA_IDS}")

    ensure_camera_loader_running()

    with jobs_lock:
        existing = jobs.get(camera_id)
        if existing and existing.thread and existing.thread.is_alive() and existing.status in ("starting", "running", "stopping"):
            return JSONResponse(
                status_code=409,
                content={
                    "status": "busy",
                    "message": "Selected conveyer is already in use",
                    "camera_id": camera_id
                },
            )

        job = JobState(camera_id=camera_id, truck_visit_id=truck_visit_id)
        job.status = "starting"
        job.started_at = datetime.utcnow().isoformat()
        job.last_log = "starting worker"
        job.stop_event.clear()

        t = threading.Thread(target=camera_job_worker, args=(job,), daemon=True)
        job.thread = t
        jobs[camera_id] = job

        logger.info("[%s] API START accepted tv_id=%s", camera_id, truck_visit_id)

        t.start()

    return JSONResponse({"status": "started", "camera_id": camera_id})


# --------------------------------------------------
# API: STOP (stop one camera)
# --------------------------------------------------
@app.post("/process_packmat_end")
async def process_packmat_end(request: Request):
    data = await request.json()
    if not data or "Conveyr_id" not in data:
        raise HTTPException(status_code=400, detail="Missing Conveyr_id")

    camera_id = str(data["Conveyr_id"])

    with jobs_lock:
        job = jobs.get(camera_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"No job found for camera_id={camera_id}")

        job.stop_event.set()
        if job.status not in ("completed", "error"):
            job.status = "stopping"
        job.last_log = "stop requested"

        snap = job_public(job)

    logger.info("[%s] API STOP requested", camera_id)

    return JSONResponse(
        status_code=202,
        content={
            "status": "stopping",
            "camera_id": camera_id,
            "object_count": snap.get("count"),
            "output_path": snap.get("output_path"),
            "video_link": snap.get("video_link")
        },
    )


# --------------------------------------------------
# API: STATUS (all or one)
# --------------------------------------------------
@app.get("/process_packmat_status")
async def process_packmat_status(camera_id: Optional[str] = None):
    with jobs_lock:
        if camera_id is not None:
            cam = str(camera_id)
            job = jobs.get(cam)
            if not job:
                raise HTTPException(status_code=404, detail=f"No job found for camera_id={cam}")
            return job_public(job)

        return {cid: job_public(job) for cid, job in jobs.items()}


# --------------------------------------------------
# Health
# --------------------------------------------------
@app.get("/health")
async def health():
    active = []
    try:
        if camera_loader:
            for cid, frames in camera_loader.frame_set.items():
                if frames:
                    active.append(cid)
    except Exception:
        pass

    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "active_cameras_with_frames": active,
        "running_jobs": list(jobs.keys()),
    }


# --------------------------------------------------
# Local run
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "index:app",  # change if your filename is different
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5005)),
        log_level="info",
    )


