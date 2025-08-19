from flask import Flask, request, jsonify
from get_rtsp_link import get_rtsp_link
from save_to_DB import save_video_log
from video_tracker import mark_video_as_processed
from packmat_counter import VideoProcessor
from video_recorder import record_camera_stream
import threading
import os

from datetime import datetime

app = Flask(__name__)

# Shared state
processing_status = {
    "status": "idle",
    "count": 0,
    "output_path": None,
    "camera_id": None
}
processing_thread = None
stop_processing = False


def concurrent_record_and_process(rtsp_link, camera_id, truck_visit_id):
    global processing_status, stop_processing

    # Start recording in its own thread
    def record():
        print(f"[{camera_id}] Starting recording...")
        record_camera_stream(camera_id, rtsp_link, duration=120)
        print(f"[{camera_id}] Recording finished.")

    # Start detection/processing in its own thread
    def detect():
        print(f"[{camera_id}] Starting object detection...")
        processor = VideoProcessor(
            video_path=rtsp_link,  # pass RTSP stream directly
            model_path="packmat_i2.pt",
            camera_id=camera_id
        )
        count = processor.process_video(stop_flag=lambda: stop_processing)
        processing_status["count"] = count
        processing_status["output_path"] = processor.output_path

        if not stop_processing:
            save_video_log(truck_visit_id, processor.output_path, count)
            mark_video_as_processed(processor.output_path)
            processing_status["status"] = "completed"
        else:
            processing_status["status"] = "stopped"
        print(f"[{camera_id}] Detection finished.")

    # Run both tasks concurrently
    recorder_thread = threading.Thread(target=record)
    processor_thread = threading.Thread(target=detect)

    recorder_thread.start()
    processor_thread.start()

    recorder_thread.join()
    processor_thread.join()


@app.route("/process_packmat", methods=["POST"])
def process_video_and_generate_output():
    global processing_thread, stop_processing, processing_status

    data = request.get_json()
    if not data or "trigger" not in data or "Conveyr_id" not in data or "truck_visit_id" not in data:
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
            return jsonify({
                "status": "error",
                "message": f"No RTSP link found for camera ID {camera_id}."
            }), 404
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    processing_status.update({
        "status": "running",
        "count": 0,
        "output_path": None,
        "camera_id": camera_id
    })

    stop_processing = False

    processing_thread = threading.Thread(
        target=concurrent_record_and_process,
        args=(rtsp_link, camera_id, truck_visit_id)
    )
    processing_thread.start()

    return jsonify({
        "status": "started",
        "message": "Recording and processing started concurrently.",
        "camera_id": camera_id
    }), 200


@app.route("/process_packmat_end", methods=["POST"])
def stop_and_return_count():
    global stop_processing, processing_thread, processing_status

    if processing_status["status"] == "running":
        stop_processing = True
        if processing_thread and processing_thread.is_alive():
            processing_thread.join(timeout=10)

        return jsonify({
            "status": "stopped",
            "message": "Stopped manually.",
            "object_count": processing_status["count"],
            "output_path": processing_status["output_path"]
        }), 200

    elif processing_status["status"] == "completed":
        return jsonify({
            "status": "completed",
            "object_count": processing_status["count"],
            "output_path": processing_status["output_path"]
        }), 200

    else:
        return jsonify({
            "status": "idle",
            "message": "No processing running."
        }), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5005)


