import argparse
import logging
import time
from collections import deque

import cv2
import numpy as np

from deep_rppg_demo import draw_face_box, import_rppg, normalize_frame, parse_hr_result
from realtime_rppg_demo import (
    DISPLAY_HOLD_SECONDS,
    ESTIMATE_INTERVAL_S,
    HR_MAX_HZ,
    HR_MIN_HZ,
    MIN_DISPLAY_CONFIDENCE,
    MIN_ESTIMATE_SAMPLES,
    TARGET_FS,
    draw_plot,
    estimate_hr_from_samples,
    uniform_resample,
)


BENCH_MIN_POS_CONFIDENCE = 0.05
HR_UPDATE_INTERVAL_SECONDS = 1.0
MIN_DEEP_SIGNAL_SECONDS = 6.0
FACE_LOST_CLEAR_SECONDS = 1.5


def parse_args():
    parser = argparse.ArgumentParser(description="Compare POS and deep-learning rPPG HR estimates.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--model", default="FacePhys.rlap", help="open-rppg model name.")
    parser.add_argument("--window", type=float, default=12.0, help="POS signal window in seconds.")
    parser.add_argument("--hr-window", type=float, default=10.0, help="Deep model HR window in seconds.")
    parser.add_argument("--width", type=int, default=640, help="Requested preview width.")
    parser.add_argument("--height", type=int, default=480, help="Requested preview height.")
    parser.add_argument("--min-confidence", type=float, default=BENCH_MIN_POS_CONFIDENCE, help="Minimum POS confidence.")
    parser.add_argument("--min-sqi", type=float, default=0.20, help="Minimum deep-model SQI.")
    parser.add_argument("--hold", type=float, default=DISPLAY_HOLD_SECONDS, help="Seconds to keep last reliable BPM.")
    parser.add_argument("--warmup", type=float, default=MIN_DEEP_SIGNAL_SECONDS, help="Seconds before deep HR estimation.")
    parser.add_argument("--verbose-rppg", action="store_true", help="Show open-rppg warning logs.")
    return parser.parse_args()


def empty_result():
    return {
        "bpm": None,
        "quality": None,
        "raw_bpm": None,
        "raw_quality": None,
        "last_reliable_time": 0.0,
    }


def update_result(result, bpm, quality, minimum, hold_seconds, now):
    result["raw_bpm"] = bpm
    result["raw_quality"] = quality
    if bpm is not None and (quality is None or quality >= minimum):
        result["bpm"] = bpm
        result["quality"] = quality
        result["last_reliable_time"] = now
    elif now - result["last_reliable_time"] > hold_seconds:
        result["bpm"] = None
        result["quality"] = None


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
    return x1, y1, x2 - x1, y2 - y1


def roi_mask_for_box(frame_shape, face):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    if face is None:
        return mask, None

    x, y, w, h = face
    rects = [
        (x + int(0.30 * w), y + int(0.12 * h), int(0.40 * w), int(0.16 * h)),
        (x + int(0.18 * w), y + int(0.45 * h), int(0.25 * w), int(0.22 * h)),
        (x + int(0.57 * w), y + int(0.45 * h), int(0.25 * w), int(0.22 * h)),
    ]

    clipped = []
    fh, fw = frame_shape[:2]
    for rx, ry, rw, rh in rects:
        x1, y1 = max(0, rx), max(0, ry)
        x2, y2 = min(fw, rx + rw), min(fh, ry + rh)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
            clipped.append((x1, y1, x2 - x1, y2 - y1))
    return mask, clipped


def scale_rects(rects, source_shape, target_shape):
    if not rects:
        return rects
    source_h, source_w = source_shape[:2]
    target_h, target_w = target_shape[:2]
    scale_x = target_w / max(1, source_w)
    scale_y = target_h / max(1, source_h)
    return [
        (
            int(round(x * scale_x)),
            int(round(y * scale_y)),
            int(round(w * scale_x)),
            int(round(h * scale_y)),
        )
        for x, y, w, h in rects
    ]


def mean_rgb(frame_rgb, mask):
    pixels = frame_rgb[mask > 0]
    if len(pixels) < 64:
        return None
    return np.mean(pixels, axis=0)


def draw_metric(canvas, x, y, title, result, quality_label, minimum, color):
    bpm = result["bpm"]
    quality = result["quality"]
    raw_bpm = result["raw_bpm"]
    raw_quality = result["raw_quality"]

    bpm_text = "--" if bpm is None else f"{bpm:5.1f}"
    quality_text = "--" if quality is None else f"{quality:0.2f}"
    raw_bpm_text = "--" if raw_bpm is None else f"{raw_bpm:5.1f}"
    raw_quality_text = "--" if raw_quality is None else f"{raw_quality:0.2f}"
    bar_value = 0.0 if quality is None else float(np.clip(quality, 0.0, 1.0))

    cv2.putText(canvas, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 225, 230), 1)
    cv2.putText(canvas, bpm_text, (x, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 1.55, color, 3)
    cv2.putText(canvas, "BPM", (x + 170, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 1)
    cv2.putText(
        canvas,
        f"{quality_label}: {quality_text}  min {minimum:0.2f}",
        (x, y + 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 188, 198),
        1,
    )
    cv2.rectangle(canvas, (x, y + 88), (x + 250, y + 102), (70, 70, 70), 1)
    cv2.rectangle(canvas, (x, y + 88), (x + int(250 * bar_value), y + 102), color, -1)
    cv2.putText(
        canvas,
        f"raw {raw_bpm_text} BPM  {quality_label} {raw_quality_text}",
        (x, y + 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (150, 155, 164),
        1,
    )


def draw_dashboard(
    frame,
    roi_rects,
    pos_result,
    deep_result,
    signal_result,
    fps,
    latency,
    model_name,
    window_seconds,
    hr_window,
    min_confidence,
    min_sqi,
):
    frame_h, frame_w = frame.shape[:2]
    panel_w = 470
    canvas_h = max(frame_h, 780)
    canvas = np.zeros((canvas_h, frame_w + panel_w, 3), dtype=np.uint8)
    canvas[:frame_h, :frame_w] = frame

    panel_x = frame_w
    canvas[:, panel_x:] = (24, 26, 29)
    px = panel_x + 24

    if roi_rects:
        for x, y, w, h in roi_rects:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (90, 210, 250), 1)

    cv2.putText(canvas, "POS vs Deep rPPG", (px, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (245, 245, 245), 2)
    info_lines = [
        f"Model: {model_name}",
        f"Camera FPS: {fps:4.1f}",
        f"POS window: {window_seconds:4.1f}s",
        f"Deep window: {hr_window:4.1f}s",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(canvas, line, (px, 82 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (190, 205, 220), 1)

    draw_metric(canvas, px, 200, "POS", pos_result, "conf", min_confidence, (80, 220, 140))
    draw_metric(canvas, px, 350, "Deep Learning", deep_result, "SQI", min_sqi, (90, 210, 250))

    pos_bpm = pos_result["bpm"]
    deep_bpm = deep_result["bpm"]
    delta_text = "--" if pos_bpm is None or deep_bpm is None else f"{abs(pos_bpm - deep_bpm):0.1f} BPM"
    latency_text = "--" if latency is None else f"{latency:0.3f}s"
    cv2.putText(canvas, f"Abs diff: {delta_text}", (px, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 225, 230), 1)
    cv2.putText(canvas, f"Deep latency: {latency_text}", (px, 538), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (165, 170, 178), 1)

    plot_w = panel_w - 48
    plot_h = 70
    signal = signal_result["signal"]
    freqs = signal_result["freqs"]
    spec = signal_result["spec"]
    draw_plot(canvas, px, 585, plot_w, plot_h, signal, (80, 220, 140), "POS waveform")

    spectrum_values = None
    peak_line = None
    if freqs is not None and spec is not None:
        mask = (freqs >= HR_MIN_HZ) & (freqs <= HR_MAX_HZ)
        if np.any(mask):
            spectrum_values = spec[mask]
            if pos_bpm is not None:
                peak_line = ((pos_bpm / 60.0) - HR_MIN_HZ) / (HR_MAX_HZ - HR_MIN_HZ)
    draw_plot(canvas, px, 680, plot_w, plot_h, spectrum_values, (120, 170, 255), "POS spectrum", peak_line)

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

    rgb_history = deque()
    time_history = deque()
    pos_result = empty_result()
    deep_result = empty_result()
    signal_result = {"signal": None, "freqs": None, "spec": None}
    last_face_seen_time = 0.0
    last_pos_update = 0.0
    last_deep_update = 0.0
    last_frame_time = time.perf_counter()
    fps_smooth = 0.0
    latency = None

    cv2.namedWindow("POS vs Deep rPPG", cv2.WINDOW_NORMAL)

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
                face = parse_box(box, source_shape)
                if face is not None:
                    last_face_seen_time = now
                elif now - last_face_seen_time > FACE_LOST_CLEAR_SECONDS:
                    rgb_history.clear()
                    time_history.clear()
                    pos_result = empty_result()
                    deep_result = empty_result()
                    signal_result = {"signal": None, "freqs": None, "spec": None}

                mask, roi_rects = roi_mask_for_box(source_shape, face)
                rgb = mean_rgb(frame, mask)
                if rgb is not None:
                    rgb_history.append(rgb)
                    time_history.append(now)

                while time_history and now - time_history[0] > args.window:
                    time_history.popleft()
                    rgb_history.popleft()

                if len(rgb_history) >= MIN_ESTIMATE_SAMPLES and now - last_pos_update >= ESTIMATE_INTERVAL_S:
                    samples, fs = uniform_resample(np.array(time_history), np.array(rgb_history), target_fs=TARGET_FS)
                    signal, freqs, spec, bpm, confidence = estimate_hr_from_samples(samples, fs, "pos")
                    signal_result = {"signal": signal, "freqs": freqs, "spec": spec}
                    update_result(pos_result, bpm, confidence, args.min_confidence, args.hold, now)
                    last_pos_update = now

                if now - last_deep_update >= HR_UPDATE_INTERVAL_SECONDS:
                    signal_seconds = float(getattr(model, "now", 0.0) or 0.0)
                    if signal_seconds >= max(2.0, min(args.hr_window, args.warmup)):
                        try:
                            result = model.hr(start=-args.hr_window)
                        except Exception:
                            result = None
                        deep_bpm, deep_sqi, latency = parse_hr_result(result)
                        update_result(deep_result, deep_bpm, deep_sqi, args.min_sqi, args.hold, now)
                    last_deep_update = now

                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if args.width > 0 and args.height > 0:
                    display_frame = cv2.resize(display_frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
                draw_face_box(display_frame, box, source_shape)
                display_roi_rects = scale_rects(roi_rects, source_shape, display_frame.shape)

                dashboard = draw_dashboard(
                    display_frame,
                    display_roi_rects,
                    pos_result,
                    deep_result,
                    signal_result,
                    fps_smooth,
                    latency,
                    args.model,
                    args.window,
                    args.hr_window,
                    args.min_confidence,
                    args.min_sqi,
                )
                cv2.imshow("POS vs Deep rPPG", dashboard)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
