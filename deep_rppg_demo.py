import argparse
import logging
import platform
import sys
import time

import cv2
import numpy as np


MIN_DISPLAY_SQI = 0.20
DISPLAY_HOLD_SECONDS = 5.0
HR_UPDATE_INTERVAL_SECONDS = 1.0
MIN_HR_SIGNAL_SECONDS = 6.0


def parse_args():
    parser = argparse.ArgumentParser(description="Mac-friendly deep rPPG demo using open-rppg.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--model", default="FacePhys.rlap", help="open-rppg model name.")
    parser.add_argument("--hr-window", type=float, default=10.0, help="Seconds of recent signal used for HR.")
    parser.add_argument("--width", type=int, default=640, help="Requested preview width.")
    parser.add_argument("--height", type=int, default=480, help="Requested preview height.")
    parser.add_argument("--min-sqi", type=float, default=MIN_DISPLAY_SQI, help="Minimum SQI/confidence to update BPM.")
    parser.add_argument("--hold", type=float, default=DISPLAY_HOLD_SECONDS, help="Seconds to keep last reliable BPM.")
    parser.add_argument("--warmup", type=float, default=MIN_HR_SIGNAL_SECONDS, help="Seconds of signal to collect before HR estimation.")
    parser.add_argument("--verbose-rppg", action="store_true", help="Show open-rppg warning logs.")
    return parser.parse_args()


def import_rppg():
    try:
        import rppg
    except ImportError as exc:
        if getattr(exc, "name", None) == "pkg_resources":
            raise RuntimeError(
                "open-rppg was found, but it still imports the legacy 'pkg_resources' module. "
                "Install a setuptools version that still provides it:\n\n"
                "  python -m pip install 'setuptools<81'\n\n"
                "Then retry: python deep_rppg_demo.py"
            ) from exc
        raise RuntimeError(
            "open-rppg is not installed. Install it with: pip install -r requirements-deep.txt"
        ) from exc
    except RuntimeError as exc:
        message = str(exc)
        if "AVX" in message and "jaxlib" in message:
            machine = platform.machine()
            raise RuntimeError(
                "JAX/JAXLIB is installed with an incompatible x86/AVX build. "
                "On Apple Silicon, use a native ARM64 Python environment and reinstall JAX/open-rppg.\n\n"
                f"Current Python: {sys.executable}\n"
                f"platform.machine(): {machine}\n\n"
                "Suggested fix:\n"
                "  # Use an ARM64 Miniforge/Miniconda or native ARM64 Python first.\n"
                "  conda create -n rppg-arm python=3.10\n"
                "  conda activate rppg-arm\n"
                "  python -m pip install --upgrade pip\n"
                "  python -m pip install -r requirements-deep.txt\n\n"
                "If platform.machine() is x86_64 on an M-series Mac, your terminal or Python is running under Rosetta. "
                "Install/use an ARM64 Python distribution before reinstalling dependencies."
            ) from exc
        raise
    return rppg


def extract_metric(result, *names):
    if not isinstance(result, dict):
        return None
    for name in names:
        value = result.get(name)
        if value is not None:
            return value
    return None


def to_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def parse_hr_result(result):
    bpm = to_float(extract_metric(result, "hr", "bpm", "HR"))
    sqi = to_float(extract_metric(result, "SQI", "sqi", "confidence", "conf"))
    latency = to_float(extract_metric(result, "latency", "inference_latency", "time"))
    return bpm, sqi, latency


def reliable_result(bpm, sqi, min_sqi):
    if bpm is None:
        return False
    if sqi is None:
        return True
    return sqi >= min_sqi


def normalize_frame(frame):
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3:
        return None
    return np.ascontiguousarray(frame)


def draw_face_box(frame, box, source_shape):
    if box is None:
        return

    try:
        arr = np.asarray(box).astype(int)
    except (TypeError, ValueError):
        return

    if arr.shape == (2, 2):
        y1, y2 = arr[0]
        x1, x2 = arr[1]
    elif arr.size >= 4:
        x1, y1, x2, y2 = arr.reshape(-1)[:4]
    else:
        return

    source_h, source_w = source_shape[:2]
    target_h, target_w = frame.shape[:2]
    scale_x = target_w / max(1, source_w)
    scale_y = target_h / max(1, source_h)
    x1 = int(round(x1 * scale_x))
    x2 = int(round(x2 * scale_x))
    y1 = int(round(y1 * scale_y))
    y2 = int(round(y2 * scale_y))

    h, w = frame.shape[:2]
    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 220, 140), 2)


def draw_dashboard(
    frame,
    model_name,
    bpm,
    sqi,
    raw_bpm,
    raw_sqi,
    fps,
    latency,
    hr_window,
    min_sqi,
):
    frame_h, frame_w = frame.shape[:2]
    panel_w = 430
    canvas_h = max(frame_h, 560)
    canvas = np.zeros((canvas_h, frame_w + panel_w, 3), dtype=np.uint8)
    canvas[:frame_h, :frame_w] = frame

    panel_x = frame_w
    canvas[:, panel_x:] = (24, 26, 29)
    px = panel_x + 24

    cv2.putText(canvas, "Deep rPPG Demo", (px, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (245, 245, 245), 2)
    info_lines = [
        f"Model: {model_name}",
        f"Camera FPS: {fps:4.1f}",
        f"HR window: {hr_window:4.1f}s",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(canvas, line, (px, 82 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (190, 205, 220), 1)

    bpm_text = "--" if bpm is None else f"{bpm:5.1f}"
    cv2.putText(canvas, bpm_text, (px, 210), cv2.FONT_HERSHEY_SIMPLEX, 2.35, (80, 220, 140), 4)
    cv2.putText(canvas, "Deep BPM", (px + 210, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (210, 210, 210), 2)

    sqi_text = "--" if sqi is None else f"{sqi:0.2f}"
    sqi_bar = 0.0 if sqi is None else float(np.clip(sqi, 0.0, 1.0))
    cv2.putText(
        canvas,
        f"SQI: {sqi_text}  min {min_sqi:0.2f}",
        (px, 248),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        1,
    )
    cv2.rectangle(canvas, (px, 262), (px + 250, 278), (70, 70, 70), 1)
    cv2.rectangle(canvas, (px, 262), (px + int(250 * sqi_bar), 278), (80, 180, 120), -1)

    raw_bpm_text = "--" if raw_bpm is None else f"{raw_bpm:5.1f}"
    raw_sqi_text = "--" if raw_sqi is None else f"{raw_sqi:0.2f}"
    latency_text = "--" if latency is None else f"{latency:0.3f}s"
    cv2.putText(canvas, f"raw: {raw_bpm_text} BPM  SQI {raw_sqi_text}", (px, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (165, 170, 178), 1)
    cv2.putText(canvas, f"Latency: {latency_text}", (px, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (165, 170, 178), 1)

    help_lines = [
        "q quit",
        "BPM updates every 1s",
        "Keep face still and well lit.",
        "Research demo, not medical.",
    ]
    for i, line in enumerate(help_lines):
        cv2.putText(canvas, line, (px, canvas_h - 104 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (165, 170, 178), 1)

    return canvas


def main():
    args = parse_args()
    rppg = import_rppg()
    if not args.verbose_rppg:
        logging.getLogger("open-rppg").setLevel(logging.ERROR)

    try:
        model = rppg.Model(args.model)
    except Exception as exc:
        raise RuntimeError(f"Could not initialize open-rppg model '{args.model}': {exc}") from exc

    displayed_bpm = None
    displayed_sqi = None
    raw_bpm = None
    raw_sqi = None
    latency = None
    last_reliable_time = 0.0
    last_hr_update = 0.0
    last_frame_time = time.perf_counter()
    fps_smooth = 0.0

    cv2.namedWindow("Deep rPPG Demo", cv2.WINDOW_NORMAL)

    try:
        with model.video_capture(args.camera):
            for frame, box in model.preview:
                now = time.perf_counter()
                dt = max(1e-6, now - last_frame_time)
                last_frame_time = now
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt) if fps_smooth else 1.0 / dt

                frame = normalize_frame(frame)
                if frame is None:
                    continue
                source_shape = frame.shape
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if args.width > 0 and args.height > 0:
                    frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
                draw_face_box(frame, box, source_shape)

                if now - last_hr_update >= HR_UPDATE_INTERVAL_SECONDS:
                    signal_seconds = float(getattr(model, "now", 0.0) or 0.0)
                    if signal_seconds < max(2.0, min(args.hr_window, args.warmup)):
                        result = None
                    else:
                        try:
                            result = model.hr(start=-args.hr_window)
                        except Exception:
                            result = None
                    raw_bpm, raw_sqi, latency = parse_hr_result(result)
                    if reliable_result(raw_bpm, raw_sqi, args.min_sqi):
                        displayed_bpm = raw_bpm
                        displayed_sqi = raw_sqi
                        last_reliable_time = now
                    elif now - last_reliable_time > args.hold:
                        displayed_bpm = None
                        displayed_sqi = None
                    last_hr_update = now

                dashboard = draw_dashboard(
                    frame,
                    args.model,
                    displayed_bpm,
                    displayed_sqi,
                    raw_bpm,
                    raw_sqi,
                    fps_smooth,
                    latency,
                    args.hr_window,
                    args.min_sqi,
                )
                cv2.imshow("Deep rPPG Demo", dashboard)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
