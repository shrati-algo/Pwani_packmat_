#!/usr/bin/env python3

import os
import time
import logging
import threading
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from get_rtsp_link import get_rtsp_link
from save_to_DB import save_video_log
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

_file = logging.FileHandler(LOG_FILE)
_file.setFormatter(_fmt)
logger.addHandler(_file)

_console = logging.StreamHandler()
_console.setFormatter(_fmt)
logger.addHandler(_console)

# --------------------------------------------------
# FastAPI
# --------------------------------------------------
app = FastAPI(title="Packmat Service")

# --------------------------------------------------
# Shared state
# --------------------------------------------------
processing_status = {
    "status": "idle",
    "count": 0,
    "output_path": None,
    "camera_id": None,
    "video_link": None,
    "started_at": None,
    "last_log": None,
}

status_lock = threading.Lock()
processing_thread: Optional[threading.Thread] = None
stop_event = threading.Event()

# --------------------------------------------------
# Worker
# --------------------------------------------------
def frame_processing_worker(rtsp_link: str, camera_id: str, tv_id: Optional[str]):
    camera = None
    processor = None
    video_link = None

    try:
        logger.info("[%s] Worker START", camera_id)
        logger.info("[%s] RTSP link: %s", camera_id, rtsp_link)

        # Open camera
        camera = CameraLoader(rtsp_link).start()
        time.sleep(0.5)

        # Init processor
        processor = VideoProcessor(
            model_path="packmat_i2.pt",
            camera_id=camera_id,
            fps=20,
            frame_skip=2
        )

        last_frame_ts = time.time()
        frames_seen = 0
        loops = 0

        while not stop_event.is_set():
            loops += 1

            frame = camera.get_latest_frame()

            if frame is None:
                # If no frames for long time, log it
                if time.time() - last_frame_ts > 5:
                    logger.warning("[%s] No frames received for >5s", camera_id)
                    last_frame_ts = time.time()
                time.sleep(0.01)
                continue

            frames_seen += 1
            last_frame_ts = time.time()

            # Run processing
            try:
                result = processor.process_frame(frame)
            except Exception:
                logger.exception("[%s] process_frame crashed", camera_id)
                with status_lock:
                    processing_status["status"] = "error"
                    processing_status["last_log"] = "process_frame crashed"
                break

            # Unpack
            if isinstance(result, tuple):
                count, output_path = result
            else:
                count = result
                output_path = processor.output_path

            # Compute video link (filename fixed at init)
            if output_path:
                video_name = os.path.basename(output_path)
                video_link = f"http://192.168.5.82:5009/{video_name}"
            else:
                video_link = None

            # Update shared status
            if loops % 10 == 0:  # reduce lock contention
                with status_lock:
                    processing_status["count"] = count
                    processing_status["output_path"] = output_path
                    processing_status["video_link"] = video_link
                    processing_status["last_log"] = f"frames_seen={frames_seen}"

            time.sleep(0.001)

        # --------------------------------------------------
        # STOP requested: finalize last 60 seconds video
        # --------------------------------------------------
        logger.info("[%s] Stop signal received. Finalizing last 60s video...", camera_id)

        with status_lock:
            processing_status["status"] = "stopping"
            processing_status["last_log"] = "finalizing video"

        try:
            if processor:
                processor.cleanup()  # writes last 60 seconds ONCE
        except Exception:
            logger.exception("[%s] processor.cleanup failed", camera_id)

        # Final snapshot
        with status_lock:
            processing_status["status"] = "completed"
            final_count = processing_status.get("count", 0)
            output_path = processing_status.get("output_path")
            final_video_link = processing_status.get("video_link")
            processing_status["last_log"] = "completed"

        # DB + tracking
        try:
            save_video_log(tv_id, output_path, final_count, final_video_link)
        except Exception:
            logger.exception("[%s] save_video_log failed", camera_id)

        try:
            if output_path:
                mark_video_as_processed(output_path)
        except Exception:
            logger.exception("[%s] mark_video_as_processed failed", camera_id)

        logger.info("[%s] Worker COMPLETED", camera_id)

    except Exception:
        logger.exception("[%s] Worker ERROR (outer)", camera_id)
        with status_lock:
            processing_status["status"] = "error"
            processing_status["last_log"] = "worker outer exception"

    finally:
        try:
            if camera:
                camera.stop()
        except Exception:
            pass

        logger.info("[%s] Worker EXIT", camera_id)


# --------------------------------------------------
# API: START
# --------------------------------------------------
@app.post("/process_packmat")
async def process_packmat(request: Request):
    global processing_thread

    data = await request.json()

    if not data or "trigger" not in data or "Conveyr_id" not in data or "truck_visit_id" not in data:
        raise HTTPException(status_code=400, detail="Missing required parameters")

    if data["trigger"] == 0:
        return JSONResponse({"status": "ignored", "message": "Trigger was 0"})

    camera_id = str(data["Conveyr_id"])
    truck_visit_id = str(data["truck_visit_id"])

    # Prevent multiple workers
    if processing_thread and processing_thread.is_alive():
        return JSONResponse(
            status_code=409,
            content={"status": "busy", "message": "Processing already running", "camera_id": processing_status.get("camera_id")}
        )

    rtsp_link = get_rtsp_link(camera_id)
    if not rtsp_link:
        raise HTTPException(status_code=404, detail="RTSP link not found")

    with status_lock:
        processing_status.update({
            "status": "running",
            "count": 0,
            "output_path": None,
            "camera_id": camera_id,
            "video_link": None,
            "started_at": datetime.utcnow().isoformat(),
            "last_log": "starting worker"
        })

    stop_event.clear()

    processing_thread = threading.Thread(
        target=frame_processing_worker,
        args=(rtsp_link, camera_id, truck_visit_id),
        daemon=True
    )
    processing_thread.start()

    return JSONResponse({"status": "started", "camera_id": camera_id})


# --------------------------------------------------
# API: STOP
# --------------------------------------------------
@app.post("/process_packmat_end")
async def process_packmat_end():
    stop_event.set()

    with status_lock:
        processing_status["status"] = "stopping"
        processing_status["last_log"] = "stop requested"

        return JSONResponse(
            status_code=202,
            content={
                "status": "stopping",
                "object_count": processing_status.get("count"),
                "output_path": processing_status.get("output_path"),
                "video_link": processing_status.get("video_link"),
            }
        )


# --------------------------------------------------
# API: STATUS
# --------------------------------------------------
@app.get("/process_packmat_status")
async def process_packmat_status():
    with status_lock:
        return processing_status


# --------------------------------------------------
# Health
# --------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# --------------------------------------------------
# Local run
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_fastapi:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5005)),
        log_level="info"
    )


