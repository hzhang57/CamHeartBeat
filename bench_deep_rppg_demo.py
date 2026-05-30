import argparse
import logging
import time
import traceback

import cv2
import numpy as np

from deep_rppg_demo import draw_face_box, import_rppg, normalize_frame, parse_hr_result


BASELINE_MODEL = "FacePhys.rlap"
CANDIDATE_MODEL = "PhysFormer.rlap"
MIN_DISPLAY_SQI = 0.20
DISPLAY_HOLD_SECONDS = 5.0
HR_UPDATE_INTERVAL_SECONDS = 1.0
MIN_HR_SIGNAL_SECONDS = 6.0


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two open-rppg deep models on one camera stream.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--baseline-model", default=BASELINE_MODEL, help="Baseline open-rppg model name.")
    parser.add_argument("--candidate-model", default=CANDIDATE_MODEL, help="Candidate open-rppg model name.")
    parser.add_argument("--hr-window", type=float, default=10.0, help="Seconds of recent signal used for HR.")
    parser.add_argument("--width", type=int, default=640, help="Requested preview width.")
    parser.add_argument("--height", type=int, default=480, help="Requested preview height.")
    parser.add_argument("--min-sqi", type=float, default=MIN_DISPLAY_SQI, help="Minimum SQI to update BPM.")
    parser.add_argument("--hold", type=float, default=DISPLAY_HOLD_SECONDS, help="Seconds to keep last reliable BPM.")
    parser.add_argument("--warmup", type=float, default=MIN_HR_SIGNAL_SECONDS, help="Seconds of signal to collect before HR estimation.")
    parser.add_argument("--verbose-rppg", action="store_true", help="Show open-rppg warning logs and HR exceptions.")
    return parser.parse_args()


def empty_result():
    return {
        "bpm": None,
        "sqi": None,
        "raw_bpm": None,
        "raw_sqi": None,
        "latency": None,
        "last_reliable_time": 0.0,
    }


def update_result(result, bpm, sqi, latency, min_sqi, hold_seconds, now):
    result["raw_bpm"] = bpm
    result["raw_sqi"] = sqi
    result["latency"] = latency
    if bpm is not None and (sqi is None or sqi >= min_sqi):
        result["bpm"] = bpm
        result["sqi"] = sqi
        result["last_reliable_time"] = now
    elif now - result["last_reliable_time"] > hold_seconds:
        result["bpm"] = None
        result["sqi"] = None


def parse_box(box, source_shape):
    if box is None:
        return None
    try:
        arr = np.asarray(box).astype(int)
    except (TypeError, ValueError):
        return None

    if arr.shape == (2, 2):
        y1, y2 = arr[0]
        x1, x2 = arr[1]
    elif arr.size >= 4:
        x1, y1, x2, y2 = arr.reshape(-1)[:4]
    else:
        return None

    h, w = source_shape[:2]
    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_face(frame, box):
    parsed = parse_box(box, frame.shape)
    if parsed is None:
        return None
    x1, y1, x2, y2 = parsed
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return np.ascontiguousarray(face)


def safe_hr(model, hr_window, verbose):
    try:
        return model.hr(start=-hr_window)
    except Exception:
        if verbose:
            traceback.print_exc()
        return None


def maybe_update_model_result(model, result, now, args):
    signal_seconds = float(getattr(model, "now", 0.0) or 0.0)
    if signal_seconds < max(2.0, min(args.hr_window, args.warmup)):
        return
    hr_result = safe_hr(model, args.hr_window, args.verbose_rppg)
    bpm, sqi, latency = parse_hr_result(hr_result)
    update_result(result, bpm, sqi, latency, args.min_sqi, args.hold, now)


def draw_metric(canvas, x, y, title, result, color, min_sqi):
    bpm = result["bpm"]
    sqi = result["sqi"]
    raw_bpm = result["raw_bpm"]
    raw_sqi = result["raw_sqi"]
    latency = result["latency"]

    bpm_text = "--" if bpm is None else f"{bpm:5.1f}"
    sqi_text = "--" if sqi is None else f"{sqi:0.2f}"
    raw_bpm_text = "--" if raw_bpm is None else f"{raw_bpm:5.1f}"
    raw_sqi_text = "--" if raw_sqi is None else f"{raw_sqi:0.2f}"
    latency_text = "--" if latency is None else f"{latency:0.3f}s"
    sqi_bar = 0.0 if sqi is None else float(np.clip(sqi, 0.0, 1.0))

    cv2.putText(canvas, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 225, 230), 1)
    cv2.putText(canvas, bpm_text, (x, y + 54), cv2.FONT_HERSHEY_SIMPLEX, 1.65, color, 3)
    cv2.putText(canvas, "BPM", (x + 185, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 1)
    cv2.putText(canvas, f"SQI: {sqi_text}  min {min_sqi:0.2f}", (x, y + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 188, 198), 1)
    cv2.rectangle(canvas, (x, y + 98), (x + 260, y + 112), (70, 70, 70), 1)
    cv2.rectangle(canvas, (x, y + 98), (x + int(260 * sqi_bar), y + 112), color, -1)
    cv2.putText(canvas, f"raw {raw_bpm_text} BPM  SQI {raw_sqi_text}", (x, y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 155, 164), 1)
    cv2.putText(canvas, f"latency {latency_text}", (x, y + 166), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 155, 164), 1)


def draw_dashboard(
    frame,
    baseline_result,
    candidate_result,
    fps,
    baseline_model,
    candidate_model,
    hr_window,
    warmup,
    min_sqi,
):
    frame_h, frame_w = frame.shape[:2]
    panel_w = 500
    canvas_h = max(frame_h, 760)
    canvas = np.zeros((canvas_h, frame_w + panel_w, 3), dtype=np.uint8)
    canvas[:frame_h, :frame_w] = frame

    panel_x = frame_w
    canvas[:, panel_x:] = (24, 26, 29)
    px = panel_x + 24

    cv2.putText(canvas, "Deep Model Bench", (px, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (245, 245, 245), 2)
    info_lines = [
        f"Baseline: {baseline_model}",
        f"Candidate: {candidate_model}",
        f"Camera FPS: {fps:4.1f}",
        f"HR window: {hr_window:4.1f}s  warmup {warmup:4.1f}s",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(canvas, line, (px, 82 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (190, 205, 220), 1)

    draw_metric(canvas, px, 205, baseline_model, baseline_result, (80, 220, 140), min_sqi)
    draw_metric(canvas, px, 430, candidate_model, candidate_result, (90, 210, 250), min_sqi)

    baseline_bpm = baseline_result["bpm"]
    candidate_bpm = candidate_result["bpm"]
    diff_text = "--" if baseline_bpm is None or candidate_bpm is None else f"{abs(baseline_bpm - candidate_bpm):0.1f} BPM"
    cv2.putText(canvas, f"Abs diff: {diff_text}", (px, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 225, 230), 1)
    cv2.putText(canvas, "q quit", (px, 694), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (165, 170, 178), 1)
    return canvas


def main():
    args = parse_args()
    rppg = import_rppg()
    if not args.verbose_rppg:
        logging.getLogger("open-rppg").setLevel(logging.ERROR)

    try:
        baseline = rppg.Model(args.baseline_model)
    except Exception as exc:
        raise RuntimeError(f"Could not initialize baseline model '{args.baseline_model}': {exc}") from exc
    try:
        candidate = rppg.Model(args.candidate_model)
    except Exception as exc:
        raise RuntimeError(f"Could not initialize candidate model '{args.candidate_model}': {exc}") from exc

    baseline_result = empty_result()
    candidate_result = empty_result()
    last_hr_update = 0.0
    last_frame_time = time.perf_counter()
    fps_smooth = 0.0

    cv2.namedWindow("Deep Model Bench", cv2.WINDOW_NORMAL)

    try:
        with candidate:
            with baseline.video_capture(args.camera):
                for frame, box in baseline.preview:
                    now = time.perf_counter()
                    dt = max(1e-6, now - last_frame_time)
                    last_frame_time = now
                    fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt) if fps_smooth else 1.0 / dt

                    frame = normalize_frame(frame)
                    if frame is None:
                        continue

                    face = crop_face(frame, box)
                    if face is None:
                        candidate.update_face(None, hasface=False)
                    else:
                        candidate.update_face(face)

                    if now - last_hr_update >= HR_UPDATE_INTERVAL_SECONDS:
                        maybe_update_model_result(baseline, baseline_result, now, args)
                        maybe_update_model_result(candidate, candidate_result, now, args)
                        last_hr_update = now

                    source_shape = frame.shape
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    if args.width > 0 and args.height > 0:
                        display_frame = cv2.resize(display_frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
                    draw_face_box(display_frame, box, source_shape)

                    dashboard = draw_dashboard(
                        display_frame,
                        baseline_result,
                        candidate_result,
                        fps_smooth,
                        args.baseline_model,
                        args.candidate_model,
                        args.hr_window,
                        args.warmup,
                        args.min_sqi,
                    )
                    cv2.imshow("Deep Model Bench", dashboard)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
