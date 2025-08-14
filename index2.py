from flask import Flask, request, jsonify
from packmat_counter_g import VideoProcessor
from get_rtsp_link import get_rtsp_link
from video_recorder import record_camera_stream
from video_tracker import mark_video_as_processed
from save_to_DB import save_video_log
import os
import threading
from datetime import datetime

app = Flask(__name__)

processor_instance = None
processing_status = {
    "status": "idle",
    "count": 0,
    "output_path": None,
    "camera_id": None,
    "recorded_paths": []
}
_processing_lock = threading.Lock()

_recorder_thread = None
_inference_thread = None
_stop_event = threading.Event()


def _recorder_worker(camera_id: str, rtsp_link: str, stop_event: threading.Event, save_dir: str = "videos"):
    os.makedirs(save_dir, exist_ok=True)
    segment_length = 120  # seconds per segment
    while not stop_event.is_set():
        try:
            # record_camera_stream will create its own timestamped filename in save_dir
            recorded_path = record_camera_stream(
                camera_id,
                rtsp_link,
                duration=segment_length,
                output_folder=save_dir
            )

            if isinstance(recorded_path, str) and os.path.exists(recorded_path):
                with _processing_lock:
                    processing_status["recorded_paths"].append(recorded_path)

        except Exception as e:
            print(f"[RECORDER] recording failed for camera {camera_id}: {e}")
            break

    print(f"[RECORDER] Stopped recorder for camera {camera_id}")


def _inference_worker(camera_id: str, rtsp_link: str, stop_event: threading.Event, model_path: str = "packmat_i2.pt"):
    global processor_instance
    print(f"[INFER] Starting inference for camera {camera_id}")
    try:
        processor_instance = VideoProcessor(
            rtsp_url=rtsp_link,
            model_path=model_path,
            camera_id=camera_id
        )
        count = processor_instance.process_video()
        output_path = processor_instance.output_path
    except Exception as e:
        print(f"[INFER] Inference error for camera {camera_id}: {e}")
        count = 0
        output_path = None

    with _processing_lock:
        processing_status["count"] = count
        processing_status["output_path"] = output_path

    print(f"[INFER] Inference stopped for camera {camera_id}. Count={count}, output={output_path}")


@app.route("/process_packmat", methods=["POST"])
def process_video_and_generate_output():
    global _recorder_thread, _inference_thread, _stop_event
    data = request.get_json()
    if not data or "trigger" not in data or "Conveyr_id" not in data or "truck_visit_id" not in data:
        return jsonify({"status": "error", "message": "Missing required parameters"}), 400

    trigger = data["trigger"]
    camera_id = str(data["Conveyr_id"])
    truck_visit_id = data["truck_visit_id"]

    if trigger == 0:
        return jsonify({"status": "stopped", "message": "Processing not triggered."}), 200

    try:
        rtsp_link = get_rtsp_link(camera_id)
        if not rtsp_link:
            return jsonify({"status": "error", "message": f"No RTSP link for camera {camera_id}"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": f"DB error: {e}"}), 500

    if processing_status["status"] == "running":
        return jsonify({"status": "error", "message": "Another session is already running."}), 409

    with _processing_lock:
        processing_status.update({
            "status": "running",
            "count": 0,
            "output_path": None,
            "camera_id": camera_id,
            "recorded_paths": []
        })

    _stop_event.clear()

    _recorder_thread = threading.Thread(
        target=_recorder_worker, args=(camera_id, rtsp_link, _stop_event, "videos"), daemon=True)
    _recorder_thread.start()

    _inference_thread = threading.Thread(
        target=_inference_worker, args=(camera_id, rtsp_link, _stop_event, "packmat_i2.pt"), daemon=True)
    _inference_thread.start()

    return jsonify({"status": "started", "message": "Recording and inference started.", "camera_id": camera_id}), 200


@app.route("/process_packmat_end", methods=["POST"])
def stop_and_return_count():
    global _stop_event, _recorder_thread, _inference_thread, processor_instance
    data = request.get_json() or {}
    truck_visit_id = data.get("truck_visit_id", None)

    if processing_status["status"] != "running":
        return jsonify({"status": processing_status["status"], "message": "No running session."}), 200

    _stop_event.set()
    if processor_instance:
        processor_instance.stop()

    if _inference_thread:
        _inference_thread.join(timeout=30)
    if _recorder_thread:
        _recorder_thread.join(timeout=30)

    final_count = processing_status.get("count", 0)
    output_path = processing_status.get("output_path", None)

    try:
        if truck_visit_id and output_path:
            save_video_log(truck_visit_id, output_path, final_count)
    except Exception as e:
        print(f"[DB] save_video_log error: {e}")

    try:
        for rp in processing_status.get("recorded_paths", []):
            if os.path.exists(rp):
                mark_video_as_processed(rp)
    except Exception as e:
        print(f"[TRACKER] mark_video_as_processed error: {e}")

    processing_status["status"] = "completed"

    return jsonify({
        "status": "completed",
        "message": "Processing stopped and finalized.",
        "object_count": final_count,
        "output_path": output_path,
        "recorded_paths": processing_status.get("recorded_paths", [])
    }), 200


if __name__ == "__main__":
    os.makedirs("videos", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5005)

