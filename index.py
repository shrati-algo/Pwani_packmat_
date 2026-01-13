#!/usr/bin/env python3

import asyncio
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
# Logging
# --------------------------------------------------
LOG_FILE = os.environ.get("SERVICE_LOG", "service.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("packmat_service")

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
    "started_at": None
}

processing_thread: Optional[threading.Thread] = None
stop_processing = False
status_lock = threading.Lock()

truck_visit_id = None

# --------------------------------------------------
# Worker
# --------------------------------------------------
def frame_processing_worker(rtsp_link: str, camera_id: str, tv_id: Optional[str]):
    global stop_processing, processing_status

    camera = None
    processor = None

    try:
        logger.info("[%s] Worker START", camera_id)

        camera = CameraLoader(rtsp_link).start()

        processor = VideoProcessor(
            model_path="packmat_i2.pt",
            camera_id=camera_id,
            fps=20,
            frame_skip=2
        )

        while not stop_processing:
            frame = camera.get_latest_frame(camera_id=camera_id)

            if frame is not None:
                result = processor.process_frame(frame)

                # ----------------------------
                # Safely unpack return
                # ----------------------------
                if isinstance(result, tuple):
                    count, output_path = result
                else:
                    count = result
                    output_path = processor.output_path

                # ----------------------------
                # Update shared status
                # ----------------------------
                with status_lock:
                    processing_status["count"] = count
                    processing_status["output_path"] = output_path

                #print("outputpath processing:", output_path)

                # ----------------------------
                # Generate video link
                # ----------------------------
                video_link = None
                if output_path:
                    video_name = os.path.basename(output_path)
                    video_link = f"http://192.168.5.82:5009/{video_name}"

                    with status_lock:
                        processing_status["video_link"] = video_link

            time.sleep(0.001)

        # -----------------------
        # Finalize
        # -----------------------
        with status_lock:
            processing_status["status"] = "completed"
            final_count = processing_status["count"]
            output_path = processing_status["output_path"]

        try:
            save_video_log(tv_id, output_path, final_count, video_link)
        except Exception:
            logger.exception("[%s] save_video_log failed", camera_id)

        try:
            if output_path:
                mark_video_as_processed(output_path)
        except Exception:
            logger.exception("[%s] mark_video_as_processed failed", camera_id)

        logger.info("[%s] Worker COMPLETED", camera_id)

    except Exception:
        logger.exception("[%s] Worker ERROR", camera_id)
        with status_lock:
            processing_status["status"] = "error"

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
    global processing_thread, stop_processing, truck_visit_id

    data = await request.json()

    if not data or "trigger" not in data or "Conveyr_id" not in data or "truck_visit_id" not in data:
        raise HTTPException(status_code=400, detail="Missing required parameters")

    if data["trigger"] == 0:
        return JSONResponse({"status": "ignored", "message": "Trigger was 0"})

    camera_id = data["Conveyr_id"]
    truck_visit_id = data["truck_visit_id"]

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
            "started_at": datetime.utcnow().isoformat()
        })

    stop_processing = False

    processing_thread = threading.Thread(
        target=frame_processing_worker,
        args=(rtsp_link, camera_id, truck_visit_id),
        daemon=True
    )
    processing_thread.start()

    return JSONResponse({
        "status": "started",
        "camera_id": camera_id
    })

# --------------------------------------------------
# API: STOP
# --------------------------------------------------
@app.post("/process_packmat_end")
async def process_packmat_end():
    global stop_processing

    stop_processing = True

    with status_lock:
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
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat()
    }

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

