import subprocess

def is_nvidia_decoder_available() -> bool:
    """
    Check if 'nvh264dec' NVIDIA hardware decoder is available in GStreamer.
    Returns True if available, False otherwise.
    """
    try:
        # Query GStreamer plugins list for nvh264dec
        result = subprocess.run(
            ["gst-inspect-1.0", "nvh264dec"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return "Decoder" in result.stdout or "nvh264dec" in result.stdout
    except FileNotFoundError:
        # gst-inspect not installed
        return False


def get_gst_pipeline(rtsp_url, drop_frames=True, latency=0):
    """
    Build a GStreamer pipeline string for OpenCV.
    Auto-selects NVIDIA hardware decode if available, otherwise CPU decode.
    """

    use_hw_decode = is_nvidia_decoder_available()

    if use_hw_decode:
        print("[INFO] Using NVIDIA hardware decoding (nvh264dec)")
        pipeline = (
            f"rtspsrc location={rtsp_url} latency={latency} protocols=tcp drop-on-latency=true ! "
            f"rtph264depay ! h264parse ! nvh264dec ! "
            f"videoconvert ! "
            f"appsink drop={1 if drop_frames else 0} sync=false max-buffers=1"
        )
    else:
        print("[INFO] NVIDIA decoder not found, using CPU decoding (avdec_h264)")
        pipeline = (
            f"rtspsrc location={rtsp_url} latency={latency} protocols=tcp drop-on-latency=true ! "
            f"rtph264depay ! h264parse ! avdec_h264 ! "
            f"videoconvert ! "
            f"appsink drop={1 if drop_frames else 0} sync=false max-buffers=1"
        )

    return pipeline


# Example usage
# if __name__ == "__main__":
#     url = "rtsp://192.168.29.29:8554/mystream"
#     gst_str = get_gst_pipeline(url, drop_frames=True, latency=0)
#     print("Generated GStreamer pipeline:\n", gst_str)
