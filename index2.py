from flask import Flask, request, jsonify
from get_rtsp_link import get_rtsp_link
from save_to_DB import save_video_log
from video_tracker import mark_video_as_processed
from packmat_counter import VideoProcessor
from video_recorder import record_camera_stream

import threading
import os
import logging
import time
from datetime import datetime

app = Flask(__name__)

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    filename="service.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("packmat_service")

# Shared state
processing_status = {
    "status": "idle",
    "count": 0,
    "output_path": None,
    "camera_id": None,
    "video_link": None
}
processing_thread = None
stop_processing = False

# Lock to protect processing_status updates
status_lock = threading.Lock()


def concurrent_record_and_process(rtsp_link, camera_id, truck_visit_id):
    """
    Worker that runs two threads: recorder and detector.
    Robust to exceptions and uses lock to update shared state.
    """
    global processing_status, stop_processing

    try:
        logger.info("[%s] Worker started for truck_visit_id=%s rtsp=%s", camera_id, truck_visit_id, rtsp_link)

        # Start recording in its own thread
        def record():
            try:
                logger.info("[%s] Starting recording...", camera_id)
                record_camera_stream(camera_id, rtsp_link, duration=120)
                logger.info("[%s] Recording finished.", camera_id)
            except Exception as e:
                logger.exception("[%s] Recording crashed: %s", camera_id, e)

        # Start detection/processing in its own thread
        def detect():
            global processing_status, stop_processing
            try:
                logger.info("[%s] Starting object detection...", camera_id)
                processor = VideoProcessor(
                    video_path=rtsp_link,  # pass RTSP stream directly
                    model_path="packmat_i2.pt",
                    camera_id=camera_id
                )
                count = processor.process_video(stop_flag=lambda: stop_processing)

                # Update shared state safely
                with status_lock:
                    processing_status["count"] = count
                    processing_status["output_path"] = getattr(processor, "output_path", None)

                video_link = None
                if getattr(processor, "output_path", None):
                    video_name = os.path.basename(processor.output_path)
                    video_link = f"http://192.168.5.82:5009/{video_name}"
                    with status_lock:
                        processing_status["video_link"] = video_link

                if not stop_processing:
                    try:
                        save_video_log(truck_visit_id, processor.output_path, count, video_link)
                    except Exception as e:
                        logger.exception("[%s] save_video_log failed: %s", camera_id, e)
                    try:
                        if processor.output_path:
                            mark_video_as_processed(processor.output_path)
                    except Exception as e:
                        logger.exception("[%s] mark_video_as_processed failed: %s", camera_id, e)

                    with status_lock:
                        processing_status["status"] = "completed"
                else:
                    with status_lock:
                        processing_status["status"] = "stopped"

                logger.info("[%s] Detection finished. count=%s output=%s", camera_id, count, processor.output_path)
            except Exception as e:
                with status_lock:
                    processing_status["status"] = "error"
                logger.exception("[%s] Detection crashed: %s", camera_id, e)

        # Run both tasks concurrently as daemon threads (so they don't block process exit)
        recorder_thread = threading.Thread(target=record, daemon=True)
        processor_thread = threading.Thread(target=detect, daemon=True)

        recorder_thread.start()
        processor_thread.start()

        # Wait with timeout so we don't block forever (defensive)
        recorder_thread.join(timeout=10)
        processor_thread.join(timeout=10)

        # If threads still alive after join timeout, we log and allow them to run or be terminated externally
        if recorder_thread.is_alive() or processor_thread.is_alive():
            logger.warning("[%s] One or more worker threads still alive after join timeout.", camera_id)

        logger.info("[%s] Worker exiting normally", camera_id)

    except Exception as e:
        with status_lock:
            processing_status["status"] = "error"
        logger.exception("[%s] Worker top-level exception: %s", camera_id, e)


@app.route("/process_packmat", methods=["POST"])
def process_video_and_generate_output():
    global processing_thread, stop_processing, processing_status
    global truck_visit_id

    data = request.get_json(silent=True)
    if not data or "trigger" not in data or "Conveyr_id" not in data or "truck_visit_id" not in data:
        logger.warning("Missing parameters in request: %s", data)
        return jsonify({
            "status": "error",
            "message": "Missing required parameters."
        }), 400

    if data["trigger"] == 0:
        return jsonify({
            "status": "stopped",
            "message": "Trigger was 0."
        }), 200

    camera_id = data["Conveyr_id"]
    truck_visit_id = data["truck_visit_id"]

    try:
        rtsp_link = get_rtsp_link(camera_id)
        if not rtsp_link:
            logger.warning("No RTSP link for camera %s", camera_id)
            return jsonify({
                "status": "error",
                "message": f"No RTSP link found for camera ID {camera_id}."
            }), 404
    except Exception as e:
        logger.exception("Error resolving RTSP link for camera %s: %s", camera_id, e)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

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

    # Start worker thread (daemon so systemd can manage process lifecycle)
    processing_thread = threading.Thread(
        target=concurrent_record_and_process,
        args=(rtsp_link, camera_id, truck_visit_id),
        daemon=True
    )
    processing_thread.start()
    logger.info("Started processing thread for camera %s", camera_id)

    return jsonify({
        "status": "started",
        "message": "Recording and processing started concurrently.",
        "camera_id": camera_id
    }), 200


@app.route("/process_packmat_end", methods=["POST"])
def stop_and_return_count():
    global stop_processing, processing_thread, processing_status

    data = request.get_json(silent=True) or {}
    # Allow caller to optionally provide truck_visit_id - defensive
    tv_id = data.get("truck_visit_id", None)

    with status_lock:
        current_status = processing_status.get("status", "idle")
        output_path = processing_status.get("output_path")
        object_count = processing_status.get("count")
        video_link = processing_status.get("video_link")

    if current_status == "running":
        logger.info("Stop requested - setting stop_processing flag")
        stop_processing = True

        # Wait briefly for worker to respect the flag
        if processing_thread and processing_thread.is_alive():
            processing_thread.join(timeout=10)
            if processing_thread.is_alive():
                logger.warning("Processing thread still alive after join timeout.")

        # Save final log defensively
        try:
            save_video_log(tv_id or truck_visit_id, output_path=output_path, counter=object_count, video_link=video_link)
        except Exception as e:
            logger.exception("save_video_log failed in stop endpoint: %s", e)

        return jsonify({
            "status": "stopped",
            "message": "Stopped manually.",
            "object_count": object_count,
            "output_path": output_path,
            "video_link": video_link
        }), 200

    elif current_status == "completed":
        try:
            save_video_log(tv_id or truck_visit_id, output_path=output_path, counter=object_count, video_link=video_link)
        except Exception as e:
            logger.exception("save_video_log failed in completed branch: %s", e)

        return jsonify({
            "status": "completed",
            "object_count": object_count,
            "output_path": output_path,
            "video_link": video_link
        }), 200

    else:
        return jsonify({
            "status": "idle",
            "message": "No processing running."
        }), 200


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5005)
