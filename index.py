from flask import Flask, request, jsonify
from packmat_counter_2 import VideoProcessor
from get_rtsp_link import get_rtsp_link
from video_recorder import record_camera_stream
from video_tracker import mark_video_as_processed
from save_to_DB import save_video_log
import os
import threading
import time

app = Flask(__name__)

# Shared state
processor_instance = None
processing_status = {
    "status": "idle",
    "count": 0,
    "output_path": None,
    "camera_id": None
}
processing_thread = None
stop_processing = False  # Flag to stop processing

@app.route("/process_packmat", methods=["POST"])
def process_video_and_generate_output():
    global processor_instance, processing_status, processing_thread, stop_processing

    data = request.get_json()
    if not data or "trigger" not in data or "Conveyr_id" not in data or "truck_visit_id" not in data:
        return jsonify({
            "status": "error",
            "message": "Missing 'trigger' or 'camera_id' or 'truck_visit_id' in request."
        }), 400

    trigger = data["trigger"]
    camera_id = data["Conveyr_id"]
    truck_visit_id = data["truck_visit_id"]

    if trigger == 0:
        return jsonify({
            "status": "stopped",
            "message": "Processing not triggered."
        }), 200

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
            "message": f"Database error: {str(e)}"
        }), 500

    try:
        processing_status["status"] = "running"
        processing_status["count"] = 0
        processing_status["output_path"] = None
        processing_status["camera_id"] = camera_id

        stop_processing = False  # Reset stop flag for this session

        def run_processor():
            global processor_instance

            recorded_video_path = record_camera_stream(camera_id, rtsp_link, duration=120)
            if not recorded_video_path or not os.path.exists(recorded_video_path):
                print(f"[{camera_id}] Recording failed.")
                processing_status["status"] = "error"
                return

            processor_instance = VideoProcessor(
                video_path=recorded_video_path,
                model_path="packmat_i2.pt",
                camera_id=camera_id
            )
            count = processor_instance.process_video()
            output_path = processor_instance.output_path

            processing_status["count"] = count
            processing_status["output_path"] = output_path

            # Save to MySQL DB-- truck id/ path and count
            save_video_log(truck_visit_id, output_path, count)

            mark_video_as_processed(recorded_video_path)
            processing_status["status"] = "completed"

        processing_thread = threading.Thread(target=run_processor)
        processing_thread.start()

        # Immediately return response without waiting for the background task to finish
        return jsonify({
            "status": "started",
            "message": "Processing started in the background.",
            "camera_id": camera_id
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Processing error: {str(e)}"
        }), 500


@app.route("/process_packmat_end", methods=["POST"])
def stop_and_return_count():
    global processing_status

    if processing_status["status"] == "completed":
        return jsonify({
            "status": "completed",
            "message": "Processing completed.",
            "object_count": processing_status["count"],
            "output_path": processing_status["output_path"]
        }), 200
    elif processing_status["status"] == "running":
        return jsonify({
            "status": "running",
            "message": "Processing is still running.",
            "object_count": processing_status["count"]
        }), 200
    else:
        return jsonify({
            "status": "idle",
            "message": "No processing is currently running.",
            "object_count": processing_status["count"]
        }), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)
