#!/usr/bin/env python3
# packmat_counter : old version we were using which gave undercounts
# packmat_counter2 the version which was performing best on test data for all classes in counting updated on 28.02.26 at 1am

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
from save_DB import save_video_log_start, update_video_log_end
from video_tracker import mark_video_as_processed
from frame_Capture import CameraLoader
from packmat_counter2 import VideoProcessor

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
# Tunables
# --------------------------------------------------
FIXED_CAMERA_IDS = [1, 2, 3, 4, 5]   # fixed cameras / conveyors
CAMERA_LOADER_THREADS = 5            # decoding workers inside CameraLoader
DEFAULT_FPS = 15

JOB_LOOP_SLEEP = 0.001
STATUS_UPDATE_EVERY_N_LOOPS = 10
NO_FRAME_WARN_SECS = 5

END_CALL_WAIT_TIMEOUT_SECS = 60

# --------------------------------------------------
# Job model
# --------------------------------------------------
@dataclass
class JobState:
    camera_id: str
    truck_visit_id: Optional[str] = None
    truck_product_visit_id: Optional[str] = None

    status: str = "idle"  # idle | starting | running | stopping | completed | error
    count: int = 0
    output_path: Optional[str] = None
    video_link: Optional[str] = None
    started_at: Optional[str] = None
    last_log: Optional[str] = None
    error: Optional[str] = None

    # DB row id created at START, used to update at END
    db_log_id: Optional[int] = None

    # capture END pressed time to write into updatedAt
    end_pressed_at: Optional[str] = None

    # debug fields
    stop_requested_by_api: bool = False
    exit_reason: Optional[str] = None
    frames_seen: int = 0

    stop_event: threading.Event = field(default_factory=threading.Event)
    completion_event: threading.Event = field(default_factory=threading.Event)

    thread: Optional[threading.Thread] = None
    processor: Optional[VideoProcessor] = None


def job_public(job: JobState) -> Dict[str, Any]:
    return {
        "camera_id": job.camera_id,
        "truck_visit_id": job.truck_visit_id,
        "truck_product_visit_id": job.truck_product_visit_id,
        "status": job.status,
        "count": job.count,
        "output_path": job.output_path,
        "video_link": job.video_link,
        "started_at": job.started_at,
        "last_log": job.last_log,
        "error": job.error,
        "db_log_id": job.db_log_id,
        "end_pressed_at": job.end_pressed_at,
        "stop_requested_by_api": job.stop_requested_by_api,
        "exit_reason": job.exit_reason,
        "frames_seen": job.frames_seen,
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
            rtsp = ""

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
            logger.info(
                "[INIT] %s -> cameraid=%s rtsp=%s",
                name,
                cfg.get("cameraid"),
                "SET" if cfg.get("rtsp") else "EMPTY"
            )

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
    p_id = job.truck_product_visit_id

    logger.info("[%s] JOB START tv_id=%s p_id=%s", cam_id, tv_id, p_id)

    exit_reason = "worker_started"

    try:
        try:
            ensure_camera_loader_running()
        except Exception as e:
            exit_reason = f"camera_loader_init_failed: {str(e)[:200]}"
            raise

        try:
            processor = VideoProcessor(
                model_path="model_logs/packmat_i2.pt",
                camera_id=cam_id,
                fps=DEFAULT_FPS
            )
        except Exception as e:
            exit_reason = f"processor_init_failed: {str(e)[:200]}"
            raise

        job.processor = processor

        with jobs_lock:
            job.status = "running"
            job.started_at = datetime.utcnow().isoformat()
            job.last_log = "running"
            job.error = None
            job.exit_reason = None

        last_frame_ts = time.time()
        frames_seen = 0
        loops = 0
        last_print = time.time()

        while not job.stop_event.is_set():
            loops += 1

            try:
                frame = camera_loader.get_latest_frame(int(cam_id))
            except Exception as e:
                exit_reason = f"get_latest_frame_exception: {str(e)[:200]}"
                logger.exception("[%s] get_latest_frame crashed", cam_id)
                with jobs_lock:
                    job.last_log = f"get_latest_frame crashed: {str(e)[:200]}"
                    job.exit_reason = exit_reason
                frame = None

            if frame is None:
                try:
                    st = getattr(camera_loader, "state", None)
                    st_val = st.get(int(cam_id)) if isinstance(st, dict) else None
                except Exception:
                    st_val = None

                logger.warning("[%s] frame None | status=%s", cam_id, st_val)

                if time.time() - last_frame_ts > NO_FRAME_WARN_SECS:
                    logger.warning("[%s] No frames received for >%ss", cam_id, NO_FRAME_WARN_SECS)
                    last_frame_ts = time.time()
                    with jobs_lock:
                        job.last_log = f"no frames >{NO_FRAME_WARN_SECS}s"

                time.sleep(0.01)
                continue

            frames_seen += 1
            last_frame_ts = time.time()

            with jobs_lock:
                job.frames_seen = frames_seen

            try:
                count, output_path = processor.process_frame(frame)
            except Exception as e:
                exit_reason = f"process_frame_exception: {str(e)[:200]}"
                logger.exception("[%s] process_frame crashed | frames_seen=%s", cam_id, frames_seen)
                with jobs_lock:
                    job.status = "error"
                    job.error = "process_frame crashed"
                    job.last_log = str(e)[:250]
                    job.exit_reason = exit_reason
                break

            if output_path:
                video_name = os.path.basename(output_path)
                video_link = f"http://192.168.5.82:5009/{video_name}"
            else:
                video_link = None

            if loops % STATUS_UPDATE_EVERY_N_LOOPS == 0:
                with jobs_lock:
                    job.count = int(count) if count is not None else job.count
                    job.output_path = output_path
                    job.video_link = video_link
                    job.last_log = f"frames_seen={frames_seen}"

            if time.time() - last_print > 5:
                logger.info("[%s] Running | frames_seen=%s | count=%s", cam_id, frames_seen, job.count)
                last_print = time.time()

            time.sleep(JOB_LOOP_SLEEP)

        if job.stop_event.is_set():
            if job.stop_requested_by_api:
                exit_reason = "stop_event_set_by_end_api"
            else:
                exit_reason = "stop_event_set_without_end_api"

        with jobs_lock:
            job.exit_reason = exit_reason

        logger.info(
            "[%s] Worker loop exited | reason=%s | stop_event=%s | api_stop=%s | frames_seen=%s",
            cam_id,
            exit_reason,
            job.stop_event.is_set(),
            job.stop_requested_by_api,
            frames_seen,
        )

        logger.info("[%s] Stop requested. Finalizing last 60s video...", cam_id)
        with jobs_lock:
            if job.status != "error":
                job.status = "stopping"
            job.last_log = "finalizing video"

        try:
            processor.cleanup()
        except Exception as e:
            exit_reason = f"cleanup_failed: {str(e)[:200]}"
            logger.exception("[%s] processor.cleanup failed", cam_id)
            with jobs_lock:
                job.status = "error"
                job.error = "processor.cleanup failed"
                job.exit_reason = exit_reason

        # Snapshot for DB update
        with jobs_lock:
            if job.status != "error":
                job.status = "completed"
                job.last_log = "completed"

            final_count = job.count
            final_output = job.output_path
            final_link = job.video_link
            log_id = job.db_log_id
            end_ts = job.end_pressed_at or datetime.utcnow().isoformat()

        # update DB row created at START
        try:
            try:
                if log_id:
                    ok = update_video_log_end(
                        log_id=log_id,
                        output_path=final_output,
                        object_count=final_count,
                        video_link=final_link,
                        end_pressed_at_iso=end_ts
                    )
                    if ok:
                        logger.info("[%s] DB log updated id=%s count=%s updatedAt=%s", cam_id, log_id, final_count, end_ts)
                    else:
                        logger.warning("[%s] DB log update returned False id=%s", cam_id, log_id)
                else:
                    logger.warning("[%s] No db_log_id present; cannot update END log", cam_id)
                    with jobs_lock:
                        job.last_log = "No db_log_id present; cannot update END log"
            except Exception as e:
                exit_reason = f"db_update_failed: {str(e)[:200]}"
                logger.exception("[%s] update_video_log_end failed", cam_id)
                with jobs_lock:
                    job.last_log = "update_video_log_end failed"
                    job.exit_reason = exit_reason
        finally:
            job.completion_event.set()

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
        exit_reason = f"outer_exception: {str(e)[:200]}"
        logger.exception("[%s] JOB OUTER ERROR", cam_id)
        with jobs_lock:
            job.status = "error"
            job.error = "job outer exception"
            job.last_log = str(e)[:250]
            job.exit_reason = exit_reason
        job.completion_event.set()

    finally:
        with jobs_lock:
            final_status = job.status
            final_reason = job.exit_reason
            final_frames = job.frames_seen
            final_api_stop = job.stop_requested_by_api

        logger.info(
            "[%s] JOB EXIT | final_status=%s | exit_reason=%s | frames_seen=%s | api_stop=%s",
            cam_id,
            final_status,
            final_reason,
            final_frames,
            final_api_stop,
        )

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
    truck_product_visit_id = str(data.get("truck_product_visit_id", ""))

    if camera_id not in {str(x) for x in FIXED_CAMERA_IDS}:
        raise HTTPException(status_code=400, detail=f"Invalid Conveyr_id={camera_id}. Allowed: {FIXED_CAMERA_IDS}")

    ensure_camera_loader_running()

    with jobs_lock:
        existing = jobs.get(camera_id)
        if (
            existing
            and existing.thread
            and existing.thread.is_alive()
            and existing.status in ("starting", "running", "stopping")
        ):
            return JSONResponse(
                status_code=409,
                content={
                    "status": "busy",
                    "message": "Selected conveyor is already in use.",
                    "camera_id": camera_id
                },
            )

        job = JobState(
            camera_id=camera_id,
            truck_visit_id=truck_visit_id,
            truck_product_visit_id=truck_product_visit_id
        )
        job.status = "starting"
        job.started_at = datetime.utcnow().isoformat()
        job.last_log = "starting worker"
        job.stop_event.clear()
        job.completion_event.clear()
        job.stop_requested_by_api = False
        job.exit_reason = None
        job.frames_seen = 0

        try:
            job.db_log_id = save_video_log_start(truck_visit_id, truck_product_visit_id)
            logger.info("[%s] START DB log inserted id=%s", camera_id, job.db_log_id)
            job.last_log = f"db start logged id={job.db_log_id}"
        except Exception:
            logger.exception("[%s] save_video_log_start failed", camera_id)
            job.last_log = "save_video_log_start failed"

        # non-daemon thread so job is treated as important worker
        t = threading.Thread(target=camera_job_worker, args=(job,))
        job.thread = t
        jobs[camera_id] = job

        logger.info("[%s] API START accepted tv_id=%s p_id=%s", camera_id, truck_visit_id, truck_product_visit_id)
        t.start()

    return JSONResponse({
        "status": "started",
        "camera_id": camera_id,
        "db_log_id": jobs.get(camera_id).db_log_id if camera_id in jobs else None
    })

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
            raise HTTPException(status_code=404, detail=f"No offloading has been started on conveyor number :{camera_id}")

        job.end_pressed_at = datetime.utcnow().isoformat()
        job.stop_requested_by_api = True
        job.stop_event.set()

        if job.status not in ("completed", "error"):
            job.status = "stopping"
        job.last_log = "stop requested, waiting for finalize+db"

    logger.info("[%s] API STOP requested — waiting for finalize+db", camera_id)

    completed = job.completion_event.wait(timeout=END_CALL_WAIT_TIMEOUT_SECS)

    if completed and job.thread and job.thread.is_alive():
        job.thread.join(timeout=2)

    with jobs_lock:
        snap = job_public(job)

    if not completed:
        logger.warning("[%s] Timeout waiting for finalize+db (returning 202)", camera_id)
        return JSONResponse(
            status_code=202,
            content={
                "status": "stopping",
                "warning": "Finalize/DB save still in progress",
                "camera_id": camera_id,
                "object_count": snap.get("count"),
                "output_path": snap.get("output_path"),
                "video_link": snap.get("video_link"),
                "db_log_id": snap.get("db_log_id"),
                "exit_reason": snap.get("exit_reason"),
                "frames_seen": snap.get("frames_seen"),
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "status": snap.get("status"),
            "camera_id": camera_id,
            "object_count": snap.get("count"),
            "output_path": snap.get("output_path"),
            "video_link": snap.get("video_link"),
            "db_log_id": snap.get("db_log_id"),
            "end_pressed_at": snap.get("end_pressed_at"),
            "exit_reason": snap.get("exit_reason"),
            "frames_seen": snap.get("frames_seen"),
            "stop_requested_by_api": snap.get("stop_requested_by_api"),
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
    running_jobs = []

    try:
        if camera_loader:
            for cid, frames in camera_loader.frame_set.items():
                if frames:
                    active.append(cid)
    except Exception:
        pass

    with jobs_lock:
        for cid, job in jobs.items():
            if (
                job.thread
                and job.thread.is_alive()
                and job.status in ("starting", "running", "stopping")
            ):
                running_jobs.append(cid)

    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "active_cameras_with_frames": active,
        "running_jobs": running_jobs,
    }

# --------------------------------------------------
# API: STATUS (available / not-running conveyors)
# --------------------------------------------------
@app.get("/packmat_status")
async def packmat_status():
    not_running = []

    with jobs_lock:
        for cid in FIXED_CAMERA_IDS:
            cam_id = str(cid)
            job = jobs.get(cam_id)

            is_running = (
                job is not None
                and job.thread is not None
                and job.thread.is_alive()
                and job.status in ("starting", "running", "stopping")
            )

            if not is_running:
                not_running.append({"id": cam_id, "name": cam_id})

    return not_running

# --------------------------------------------------
# Local run
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "index3:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5005)),
        log_level="info",
    )
