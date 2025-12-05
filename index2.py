#!/usr/bin/env python3
"""
Production-ready FastAPI conversion of your Flask app.
Keeps all original class/function names (VideoProcessor, record_camera_stream, get_rtsp_link, save_video_log, mark_video_as_processed).
Run with: gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:5005 main_fastapi:app
"""
import asyncio
import os
import time
import logging
import threading
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

# --- keep these imports exactly as in your project ---
from get_rtsp_link import get_rtsp_link
from save_to_DB import save_video_log
from video_tracker import mark_video_as_processed
from new_counter import VideoProcessor
from video_recorder import record_camera_stream

# --- Logging ---
LOG_FILE = os.environ.get("SERVICE_LOG", "service.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("packmat_service")

# --- FastAPI app ---
app = FastAPI(title="Packmat Service")

# --- Shared state (kept similar to original, but protected) ---
processing_status = {
    "status": "idle",      # idle | running | stopped | completed | error
    "count": 0,
    "output_path": None,
    "camera_id": None,
    "video_link": None,
    "started_at": None
}
processing_thread: Optional[threading.Thread] = None
stop_processing = False
status_lock = threading.Lock()

# Keep a global truck_visit_id for DB save (as original used)
truck_visit_id = None


# -----------------------------
# RTSP helper (reconnect)
# -----------------------------
def open_rtsp_with_retry(rtsp_link: str, max_attempts: int = 12, base_wait: float = 1.5):
    """
    Try to open RTSP (cv2.VideoCapture) with retries.
    Returns True when we believe stream is reachable (we do not return capture object because VideoProcessor expects rtsp).
    This function is a light pre-check. If you prefer, adapt to return a cv2.VideoCapture and use it directly.
    """
    try:
        import cv2
    except Exception:
        logger.warning("OpenCV not available for RTSP pre-check; skipping pre-check")
        return True

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        cap = cv2.VideoCapture(rtsp_link)
        opened = cap.isOpened() if cap is not None else False
        try:
            if not opened:
                logger.warning("RTSP open attempt %d/%d failed for %s", attempt, max_attempts, rtsp_link)
                try:
                    cap.release()
                except Exception:
                    pass
                wait = base_wait * (1.5 ** min(attempt - 1, 6))
                t0 = time.time()
                while time.time() - t0 < wait:
                    time.sleep(0.5)
                continue
            else:
                try:
                    cap.release()
                except Exception:
                    pass
                logger.info("RTSP pre-check succeeded for %s", rtsp_link)
                return True
        except Exception:
            try:
                cap.release()
            except Exception:
                pass
            logger.exception("Exception during RTSP pre-check")
            time.sleep(1)

    logger.error("RTSP pre-check failed after %d attempts for %s", max_attempts, rtsp_link)
    return False


# -----------------------------
# Worker: runs recorder + detection in thread
# -----------------------------
def concurrent_record_and_process(rtsp_link: str, camera_id: str, tv_id: Optional[str]):
    """
    This function runs in a background daemon thread.
    It launches two internal threads:
      - record()
      - detect()
    Updates processing_status (protected by status_lock)
    """
    global processing_status, stop_processing

    try:
        logger.info("[%s] Worker START for truck_visit_id=%s rtsp=%s", camera_id, tv_id, rtsp_link)

        # ---- recording thread ----
        def record():
            try:
                logger.info("[%s] Recorder: starting", camera_id)
                record_camera_stream(camera_id, rtsp_link, duration=120)
                logger.info("[%s] Recorder: finished", camera_id)
            except Exception as e:
                logger.exception("[%s] Recorder crashed: %s", camera_id, e)

        # ---- detection thread ----
        def detect():
            global processing_status, stop_processing
            try:
                logger.info("[%s] Detector: RTSP pre-check", camera_id)
                ok = open_rtsp_with_retry(rtsp_link)
                if not ok:
                    # Cannot connect; mark error
                    with status_lock:
                        processing_status["status"] = "error"
                    logger.error("[%s] Detector: RTSP pre-check failed, aborting detection", camera_id)
                    return

                logger.info("[%s] Detector: starting VideoProcessor", camera_id)

                # Hook that writes live count to shared dict
                def update_count_hook(latest_count):
                    try:
                        with status_lock:
                            processing_status["count"] = latest_count
                    except Exception:
                        logger.exception("update_count_hook failed")

                # create processor with hook
                processor = VideoProcessor(
                    video_path=rtsp_link,
                    model_path="packmat_i2.pt",
                    camera_id=camera_id,
                    update_hook=update_count_hook
                )


                count, output_path = processor.process_video(stop_flag=lambda: stop_processing)
                print(count)
                print(output_path)
                # final update to shared state — use returned output_path
                with status_lock:
                    processing_status["count"] = count
    
                    processing_status["output_path"] = output_path
                    print("outputpath procssing:",processing_status["output_path"])
                video_link = None
                if output_path:
                    video_name = os.path.basename(output_path)
                    video_link = f"http://192.168.5.82:5009/{video_name}"
                    with status_lock:
                        processing_status["video_link"] = video_link

                # Save results to DB only when not stopped — use returned output_path
                if not stop_processing:
                    try:
                        save_video_log(tv_id, output_path, count, video_link)
                    except Exception as e:
                        logger.exception("[%s] save_video_log failed: %s", camera_id, e)
                    try:
                        if output_path:
                            mark_video_as_processed(output_path)
                    except Exception as e:
                        logger.exception("[%s] mark_video_as_processed failed: %s", camera_id, e)

                    with status_lock:
                        processing_status["status"] = "completed"
                else:
                    with status_lock:
                        processing_status["status"] = "stopped"

                logger.info("[%s] Detector finished. count=%s output=%s", camera_id, count, output_path)

            except Exception as e:
                with status_lock:
                    processing_status["status"] = "error"
                logger.exception("[%s] Detector crashed: %s", camera_id, e)

        # ---- start both as daemon threads ----
        recorder_thread = threading.Thread(target=record, daemon=True)
        detector_thread = threading.Thread(target=detect, daemon=True)

        recorder_thread.start()
        detector_thread.start()

        # defensive joins with increased timeout
        recorder_thread.join(timeout=120)
        detector_thread.join(timeout=120)

        if recorder_thread.is_alive() or detector_thread.is_alive():
            logger.warning("[%s] One or more worker threads still alive after join timeout", camera_id)

        logger.info("[%s] Worker exiting normally", camera_id)

    except Exception as e:
        with status_lock:
            processing_status["status"] = "error"
        logger.exception("[%s] Worker top-level exception: %s", camera_id, e)


# -----------------------------
# API endpoints (FastAPI)
# -----------------------------
@app.post("/process_packmat")
async def process_video_and_generate_output(request: Request):
    """
    payload: { "trigger": 1, "Conveyr_id": "cam1", "truck_visit_id": "TV123" }
    """
    global processing_thread, stop_processing, processing_status, truck_visit_id

    data = await request.json()
    if not data or "trigger" not in data or "Conveyr_id" not in data or "truck_visit_id" not in data:
        logger.warning("Missing parameters in request: %s", data)
        raise HTTPException(status_code=400, detail="Missing required parameters.")

    if data["trigger"] == 0:
        return JSONResponse({"status": "stopped", "message": "Trigger was 0."}, status_code=200)

    camera_id = data["Conveyr_id"]
    truck_visit_id = data["truck_visit_id"]

    # get RTSP link
    try:
        rtsp_link = get_rtsp_link(camera_id)
        if not rtsp_link:
            logger.warning("No RTSP link for camera %s", camera_id)
            raise HTTPException(status_code=404, detail=f"No RTSP link found for camera ID {camera_id}.")
    except Exception as e:
        logger.exception("Error resolving RTSP link for camera %s: %s", camera_id, e)
        raise HTTPException(status_code=500, detail=str(e))

    # set status safely
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

    # Start worker thread (daemon)
    processing_thread = threading.Thread(
        target=concurrent_record_and_process,
        args=(rtsp_link, camera_id, truck_visit_id),
        daemon=True
    )
    processing_thread.start()
    logger.info("Started processing thread for camera %s (truck_visit_id=%s)", camera_id, truck_visit_id)

    return JSONResponse({"status": "started", "message": "Recording and processing started concurrently.", "camera_id": camera_id}, status_code=200)


@app.post("/process_packmat_end")
async def stop_and_return_count(request: Request):
    """
    Stop the worker thread and return final count, output_path, and video_link.
    Waits for the detection/processing thread to finish if necessary.
    """
    global stop_processing, processing_thread, processing_status, truck_visit_id

    # Signal the worker to stop
    stop_processing = True

    # Wait for the processing thread to finish (max 15 seconds)
    if processing_thread is not None:
        processing_thread.join(timeout=15)

    # Defensive: poll until output_path is set or timeout
    import time
    timeout = 10  # seconds
    start_time = time.time()
    while processing_status.get("output_path") is None and time.time() - start_time < timeout:
        await asyncio.sleep(0.2)

    # Safely read the latest status
    with status_lock:
        current_status = processing_status.get("status", "idle")
        object_count = processing_status.get("count", 0)
        output_path = processing_status.get("output_path")
        video_link = processing_status.get("video_link")

    # If output_path is still None, mark as error
    if output_path is None:
        current_status = "error"
        return JSONResponse({
            "status": current_status,
            "message": "Processing did not complete in time; output_path unavailable.",
            "object_count": object_count,
            "output_path": output_path,
            "video_link": video_link
        }, status_code=500)

    # Ensure DB save is done
    try:
        save_video_log(truck_visit_id, output_path=output_path, counter=object_count, video_link=video_link)
    except Exception as e:
        logger.exception("save_video_log failed in stop endpoint: %s", e)

    return JSONResponse({
        "status": current_status,
        "object_count": object_count,
        "output_path": output_path,
        "video_link": video_link
    }, status_code=200)
 


# Health endpoint
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat(), "processing_status": processing_status.get("status")}


# If run directly with uvicorn for local debug:
if __name__ == "__main__":
    import uvicorn
    # For local testing only. In production use gunicorn + uvicorn worker.
    uvicorn.run("main_fastapi:app", host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), log_level="info")
