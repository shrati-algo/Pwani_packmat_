#!/usr/bin/env python3
# packmat_counter : old version we were using which gave undercounts
# packmat_counter2 the version which was performing best on test data for all classes in counting updated on 28.02.26 at 1am

import os
import time
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from tinydb import TinyDB, Query
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware

from get_rtsp_link import get_rtsp_link
from save_DB import save_video_log_start, update_video_log_end
from video_tracker import mark_video_as_processed
from frame_Capture import CameraLoader
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
FIXED_CAMERA_IDS = [1, 2, 3, 4, 5]  # fixed cameras
CAMERA_LOADER_THREADS = 5           # decoding workers inside CameraLoader
DEFAULT_FPS = 15

JOB_LOOP_SLEEP = 0.001
STATUS_UPDATE_EVERY_N_LOOPS = 10
NO_FRAME_WARN_SECS = 5
END_CALL_WAIT_TIMEOUT_SECS = 60

RECOVERY_DB_PATH = os.environ.get("RECOVERY_DB_PATH", "/apps/packmat_pwani_updated/Pwani_packmat_/recovery_state2.json")
RECOVERY_HEARTBEAT_SECS = 5

# --------------------------------------------------
# Recovery store (TinyDB) - only for active jobs
# --------------------------------------------------
class RecoveryStateManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._ensure_file()
        self.db = TinyDB(db_path, storage=CachingMiddleware(JSONStorage))
        self.Job = Query()

    def _ensure_file(self):
        try:
            if not os.path.exists(self.db_path):
                with open(self.db_path, "w", encoding="utf-8") as f:
                    json.dump({}, f)
                return

            with open(self.db_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                with open(self.db_path, "w", encoding="utf-8") as f:
                    json.dump({}, f)
        except Exception:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def upsert_job(self, payload: Dict[str, Any]):
        with self.lock:
            payload["updated_at"] = datetime.utcnow().isoformat()
            self.db.upsert(payload, self.Job.camera_id == str(payload["camera_id"]))
            self.db.storage.flush()

    def get_job(self, camera_id: str):
        with self.lock:
            rows = self.db.search(self.Job.camera_id == str(camera_id))
            return rows[0] if rows else None

    def get_all_jobs(self):
        with self.lock:
            return list(self.db.all())

    def delete_job(self, camera_id: str):
        with self.lock:
            self.db.remove(self.Job.camera_id == str(camera_id))
            self.db.storage.flush()

    def close(self):
        with self.lock:
            self.db.close()


recovery_state = RecoveryStateManager(RECOVERY_DB_PATH)

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
    }


def save_job_to_recovery(job: JobState):
    payload = {
        "camera_id": str(job.camera_id),
        "truck_visit_id": job.truck_visit_id,
        "truck_product_visit_id": job.truck_product_visit_id,
        "status": job.status,
        "count": int(job.count or 0),
        "output_path": job.output_path,
        "video_link": job.video_link,
        "started_at": job.started_at,
        "last_log": job.last_log,
        "error": job.error,
        "db_log_id": job.db_log_id,
        "end_pressed_at": job.end_pressed_at,
    }
    recovery_state.upsert_job(payload)


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
# Recovery restore
# --------------------------------------------------
def restore_jobs_from_recovery():
    try:
        rows = recovery_state.get_all_jobs()
        if not rows:
            logger.info("[RECOVERY] No active jobs found")
            return

        logger.info("[RECOVERY] Found %s active jobs to restore", len(rows))

        ensure_camera_loader_running()

        for row in rows:
            cam_id = str(row.get("camera_id"))
            status = row.get("status")

            if cam_id not in {str(x) for x in FIXED_CAMERA_IDS}:
                logger.warning("[RECOVERY] Ignoring invalid camera_id=%s", cam_id)
                continue

            if status not in ("starting", "running", "stopping"):
                continue

            with jobs_lock:
                existing = jobs.get(cam_id)
                if existing and existing.thread and existing.thread.is_alive():
                    continue

                job = JobState(
                    camera_id=cam_id,
                    truck_visit_id=row.get("truck_visit_id"),
                    truck_product_visit_id=row.get("truck_product_visit_id"),
                )
                job.status = "running"
                job.count = int(row.get("count") or 0)
                job.output_path = row.get("output_path")
                job.video_link = row.get("video_link")
                job.started_at = row.get("started_at") or datetime.utcnow().isoformat()
                job.last_log = "restored from recovery db"
                job.error = row.get("error")
                job.db_log_id = row.get("db_log_id")
                job.end_pressed_at = row.get("end_pressed_at")

                job.stop_event.clear()
                job.completion_event.clear()

                t = threading.Thread(target=camera_job_worker, args=(job,), daemon=True)
                job.thread = t
                jobs[cam_id] = job
                t.start()

                logger.info("[RECOVERY] Restored conveyor %s with count=%s", cam_id, job.count)

    except Exception:
        logger.exception("[RECOVERY] Failed to restore jobs")

# --------------------------------------------------
# Per-camera job worker
# --------------------------------------------------
def camera_job_worker(job: JobState):
    cam_id = job.camera_id
    tv_id = job.truck_visit_id
    p_id = job.truck_product_visit_id

    logger.info("[%s] JOB START tv_id=%s p_id=%s", cam_id, tv_id, p_id)

    try:
        ensure_camera_loader_running()

        processor = VideoProcessor(
        model_path="model_logs/packmat_i2.pt",
        camera_id=cam_id,
        fps=DEFAULT_FPS,
        initial_count=job.count,
        output_path=job.output_path
        )
        job.processor = processor

        # if processor supports restoring count, set it
        try:
            if hasattr(processor, "object_count"):
                processor.object_count = int(job.count or 0)
        except Exception:
            logger.warning("[%s] Could not seed processor count from recovery state", cam_id)

        with jobs_lock:
            job.status = "running"
            if not job.started_at:
                job.started_at = datetime.utcnow().isoformat()
            job.last_log = "running"
            job.error = None

        save_job_to_recovery(job)

        last_frame_ts = time.time()
        frames_seen = 0
        loops = 0
        last_print = time.time()
        last_recovery_save = time.time()
        last_saved_count = int(job.count or 0)

        while not job.stop_event.is_set():
            loops += 1

            try:
                frame = camera_loader.get_latest_frame(int(cam_id))
            except Exception:
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
                    job.error=("[%s] No frames received for >%ss", cam_id, NO_FRAME_WARN_SECS)

                    last_frame_ts = time.time()
                    with jobs_lock:
                        job.last_log = f"no frames >{NO_FRAME_WARN_SECS}s"
                    save_job_to_recovery(job)

                time.sleep(0.01)
                continue

            frames_seen += 1
            last_frame_ts = time.time()

            try:
                count, output_path = processor.process_frame(frame)
            except Exception as e:
                logger.exception("[%s] process_frame crashed", cam_id)
                with jobs_lock:
                    job.status = "error"
                    job.error = "process_frame crashed"
                    job.last_log = str(e)[:250]
                save_job_to_recovery(job)
                break

            if output_path:
                video_name = os.path.basename(output_path)
                video_link = f"http://192.168.5.82:5009/{video_name}"
            else:
                video_link = None

            recovery_needs_save = False

            if loops % STATUS_UPDATE_EVERY_N_LOOPS == 0:
                with jobs_lock:
                    new_count = int(count) if count is not None else job.count
                    if new_count > int(job.count or 0):
                        job.count = new_count
                    job.output_path = output_path
                    job.video_link = video_link
                    job.last_log = f"frames_seen={frames_seen}"

                    if int(job.count or 0) > last_saved_count:
                        recovery_needs_save = True
                        last_saved_count = int(job.count or 0)

            if time.time() - last_recovery_save > RECOVERY_HEARTBEAT_SECS:
                recovery_needs_save = True
                last_recovery_save = time.time()

            if recovery_needs_save:
                save_job_to_recovery(job)

            if time.time() - last_print > 5:
                logger.info("[%s] Running | frames_seen=%s | count=%s", cam_id, frames_seen, job.count)
                last_print = time.time()

            time.sleep(JOB_LOOP_SLEEP)

        logger.info("[%s] Stop requested. Finalizing last 60s video...", cam_id)
        with jobs_lock:
            if job.status != "error":
                job.status = "stopping"
            job.last_log = "finalizing video"

        save_job_to_recovery(job)

        try:
            processor.cleanup()
        except Exception:
            logger.exception("[%s] processor.cleanup failed", cam_id)
            with jobs_lock:
                job.status = "error"
                job.error = "processor.cleanup failed"
                job.last_log = "processor.cleanup failed"
            save_job_to_recovery(job)

        # Snapshot for DB update
        with jobs_lock:
            if job.status != "error":
                job.status = "completed"
                job.last_log = "completed"

            final_count = job.count
            final_output = job.output_path
            final_link = job.video_link
            log_id = job.db_log_id
            error=job.error
            end_ts = job.end_pressed_at or datetime.utcnow().isoformat()

        # existing SQL end logging unchanged
        try:
            try:
                if log_id:
                    ok = update_video_log_end(
                        log_id=log_id,
                        output_path=final_output,
                        object_count=final_count,
                        video_link=final_link,
                        end_pressed_at_iso=end_ts,
                        error_msg= error
                    )
                    if ok:
                        logger.info("[%s] DB log updated id=%s count=%s updatedAt=%s", cam_id, log_id, final_count, end_ts)
                    else:
                        logger.warning("[%s] DB log update returned False id=%s", cam_id, log_id)
                else:
                    logger.warning("[%s] No db_log_id present; cannot update END log", cam_id)
                    with jobs_lock:
                        job.last_log = "No db_log_id present; cannot update END log"
            except Exception:
                logger.exception("[%s] update_video_log_end failed", cam_id)
                with jobs_lock:
                    job.last_log = "update_video_log_end failed"
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

        # remove from recovery db only after successful completion
        if job.status == "completed":
            try:
                recovery_state.delete_job(cam_id)
                logger.info("[%s] Removed from recovery store", cam_id)
            except Exception:
                logger.exception("[%s] Failed to remove from recovery store", cam_id)
        else:
            save_job_to_recovery(job)

        logger.info("[%s] JOB END status=%s", cam_id, job.status)

    except Exception as e:
        logger.exception("[%s] JOB OUTER ERROR", cam_id)
        with jobs_lock:
            job.status = "error"
            job.error = "job outer exception"
            job.last_log = str(e)[:250]
        save_job_to_recovery(job)
        job.completion_event.set()

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

        # existing SQL start logging unchanged
        try:
            job.db_log_id = save_video_log_start(truck_visit_id, truck_product_visit_id)
            logger.info("[%s] START DB log inserted id=%s", camera_id, job.db_log_id)
            job.last_log = f"db start logged id={job.db_log_id}"
        except Exception:
            logger.exception("[%s] save_video_log_start failed", camera_id)
            job.last_log = "save_video_log_start failed"

        # recovery only
        save_job_to_recovery(job)

        t = threading.Thread(target=camera_job_worker, args=(job,), daemon=True)
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
# Global cache for idempotent END responses
# --------------------------------------------------
completed_jobs_cache = {}  # key -> final response


@app.post("/process_packmat_end")
async def process_packmat_end(request: Request):
    client_ip = request.client.host
    headers = dict(request.headers)

    logger.info("🚨 END API CALLED 🚨")
    logger.info(f"client_ip: {client_ip}")
    logger.info(f"x-forwarded-for: {headers.get('x-forwarded-for')}")
    logger.info(f"x-real-ip: {headers.get('x-real-ip')}")
    logger.info(f"user-agent: {headers.get('user-agent')}")

    data = await request.json()

    # --------------------------------------------------
    # 0. Validate required fields
    # --------------------------------------------------
    required_fields = ["Conveyr_id", "truck_visit_id", "truck_product_visit_id", "status","sourceAPI"]

    if not data or any(field not in data for field in required_fields):
        raise HTTPException(status_code=400, detail="Missing required fields")

    camera_id = str(data["Conveyr_id"])
    truck_visit_id = str(data["truck_visit_id"])
    truck_product_visit_id = str(data["truck_product_visit_id"])
  
    incoming_status = str(data["status"]).lower()
    source_api = str(data["sourceAPI"])

    # Unique idempotency key
    job_key = f"{camera_id}_{truck_visit_id}_{truck_product_visit_id}"

    # --------------------------------------------------
    # 🔁 1. Idempotency check (return cached response)
    # --------------------------------------------------
    if job_key in completed_jobs_cache:
        logger.info(f"[{camera_id}] Returning cached END response (idempotent)")
        return JSONResponse(status_code=200, content=completed_jobs_cache[job_key])

    # --------------------------------------------------
    # 🔒 2. Allow ONLY backend to trigger END
    # --------------------------------------------------
    if incoming_status != "backend":
        logger.warning(f"[{camera_id}] Unauthorized END blocked (status={incoming_status}) via {source_api}")
        raise HTTPException(
            status_code=403,
            detail="END API can only be triggered by backend"
        )

    # --------------------------------------------------
    # 3. Try to find active job in memory
    # --------------------------------------------------
    with jobs_lock:
        job = jobs.get(camera_id)

    # --------------------------------------------------
    # 4. Try rebuilding from recovery if not in memory
    # --------------------------------------------------
    if not job:
        saved = recovery_state.get_job(camera_id)
        logger.info(f"[{camera_id}] No active job in memory, checking recovery db: {'found' if saved else 'not found'}")
        if saved:
            saved_status = saved.get("status", "running")

            if saved_status in ("running", "stopping"):
                rebuilt_job = JobState(
                    camera_id=str(saved.get("camera_id")),
                    truck_visit_id=saved.get("truck_visit_id"),
                    truck_product_visit_id=saved.get("truck_product_visit_id"),
                )
                rebuilt_job.status = saved_status
                rebuilt_job.count = int(saved.get("count") or 0)
                rebuilt_job.output_path = saved.get("output_path")
                rebuilt_job.video_link = saved.get("video_link")
                rebuilt_job.started_at = saved.get("started_at")
                rebuilt_job.last_log = "rebuilt from recovery db for stopping"
                rebuilt_job.error = saved.get("error")
                rebuilt_job.db_log_id = saved.get("db_log_id")
                rebuilt_job.end_pressed_at = saved.get("end_pressed_at")

                with jobs_lock:
                    jobs[camera_id] = rebuilt_job
                    job = rebuilt_job

    # --------------------------------------------------
    # 5. If no job anywhere → already stopped (idempotent success)
    # --------------------------------------------------
    if not job:
        logger.info(f"[{camera_id}] END called but job already completed")

        final_response = {
            "status": "completed",
            "camera_id": camera_id,
            "object_count": 0,
            "output_path": None,
            "video_link": None,
            "db_log_id": None,
            "end_pressed_at": None,
        }

        completed_jobs_cache[job_key] = final_response

        return JSONResponse(status_code=200, content=final_response)

    # --------------------------------------------------
    # 6. If job already completed → return same response
    # --------------------------------------------------
    if job.status not in ("running", "stopping"):
        with jobs_lock:
            jobs.pop(camera_id, None)

        final_response = {
            "status": job.status,
            "camera_id": camera_id,
            "object_count": job.count,
            "output_path": job.output_path,
            "video_link": job.video_link,
            "db_log_id": job.db_log_id,
            "end_pressed_at": job.end_pressed_at,
        }

        completed_jobs_cache[job_key] = final_response

        return JSONResponse(status_code=200, content=final_response)

    # --------------------------------------------------
    # 7. Request stop
    # --------------------------------------------------
    with jobs_lock:
        if not job.end_pressed_at:
            job.end_pressed_at = datetime.utcnow().isoformat()

        worker_alive = job.thread is not None and job.thread.is_alive()

        if worker_alive:
            job.stop_event.set()
            if job.status == "running":
                job.status = "stopping"
            job.last_log = "stop requested, waiting for finalize+db"
            save_job_to_recovery(job)

    logger.info("[%s] API STOP requested by %s — waiting for finalize+db", camera_id, incoming_status)

    # --------------------------------------------------
    # 8. Worker not alive case
    # --------------------------------------------------
    if not worker_alive:
        saved = recovery_state.get_job(camera_id)

        if not saved or saved.get("status") not in ("running", "stopping"):
            final_response = {
                "status": "completed",
                "camera_id": camera_id,
                "object_count": job.count,
                "output_path": job.output_path,
                "video_link": job.video_link,
                "db_log_id": job.db_log_id,
                "end_pressed_at": job.end_pressed_at,
            }

            completed_jobs_cache[job_key] = final_response

            return JSONResponse(status_code=200, content=final_response)

        return JSONResponse(
            status_code=202,
            content={
                "status": saved.get("status", job.status),
                "warning": "Stop requested earlier; finalize/DB save in progress",
                "camera_id": camera_id,
                "object_count": int(saved.get("count") or job.count or 0),
                "output_path": saved.get("output_path") or job.output_path,
                "video_link": saved.get("video_link") or job.video_link,
                "db_log_id": saved.get("db_log_id") or job.db_log_id,
                "end_pressed_at": saved.get("end_pressed_at") or job.end_pressed_at,
            },
        )

    # --------------------------------------------------
    # 9. Wait for completion
    # --------------------------------------------------
    completed = job.completion_event.wait(timeout=END_CALL_WAIT_TIMEOUT_SECS)

    with jobs_lock:
        snap = job_public(job)

    # --------------------------------------------------
    # 10. Still processing
    # --------------------------------------------------
    if not completed:
        logger.warning("[%s] Timeout waiting for finalize+db", camera_id)

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
                "end_pressed_at": snap.get("end_pressed_at"),
            },
        )

    # --------------------------------------------------
    # 11. Final response
    # --------------------------------------------------
    final_status = snap.get("status")

    final_response = {
        "status": final_status,
        "camera_id": camera_id,
        "object_count": snap.get("count"),
        "output_path": snap.get("output_path"),
        "video_link": snap.get("video_link"),
        "db_log_id": snap.get("db_log_id"),
        "end_pressed_at": snap.get("end_pressed_at"),
    }

    # Save for idempotency
    completed_jobs_cache[job_key] = final_response

    if final_status in ("completed", "error"):
        with jobs_lock:
            jobs.pop(camera_id, None)

    return JSONResponse(status_code=200, content=final_response)


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
                saved = recovery_state.get_job(cam)
                if saved:
                    return saved
                raise HTTPException(status_code=404, detail=f"No job found for camera_id={cam}")
            return job_public(job)

        all_jobs = {cid: job_public(job) for cid, job in jobs.items()}

    # include recovery-only jobs that are not loaded in memory
    try:
        recovered = recovery_state.get_all_jobs()
        for row in recovered:
            cam_id = str(row.get("camera_id"))
            if cam_id not in all_jobs:
                all_jobs[cam_id] = row
    except Exception:
        logger.exception("[STATUS] Failed to read recovery store")

    return all_jobs

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

    with jobs_lock:
        running_jobs = list(jobs.keys())

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
        memory_jobs = dict(jobs)

    for cid in FIXED_CAMERA_IDS:
        cam_id = str(cid)
        job = memory_jobs.get(cam_id)

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
# Startup / shutdown hooks
# --------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("[APP] Startup recovery begin")
    restore_jobs_from_recovery()
    logger.info("[APP] Startup recovery complete")

@app.on_event("shutdown")
async def shutdown_event():
    try:
        recovery_state.close()
    except Exception:
        logger.exception("[APP] Failed closing recovery store")

# --------------------------------------------------
# Local run
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "index3:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_level="info",
    )
    
