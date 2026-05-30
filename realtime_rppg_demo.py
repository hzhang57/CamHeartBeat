import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


HR_MIN_BPM = 45
HR_MAX_BPM = 180
HR_MIN_HZ = HR_MIN_BPM / 60.0
HR_MAX_HZ = HR_MAX_BPM / 60.0
MIN_DISPLAY_CONFIDENCE = 0.20
DISPLAY_HOLD_SECONDS = 5.0
TARGET_FS = 30.0
MIN_ESTIMATE_SAMPLES = 90
DETECTION_INTERVAL_S = 0.35
ESTIMATE_INTERVAL_S = 0.35
FACE_LOST_CLEAR_SECONDS = 1.5
POS_WINDOW_SECONDS = 1.6
POS_STEP_SECONDS = 0.1
CHROM_WINDOW_SECONDS = 1.6
CHROM_STEP_SECONDS = 0.1
RPPG_METHODS = ("pos", "chrom")
DEFAULT_YUNET_MODEL = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2022mar.onnx"


def parse_args():
    parser = argparse.ArgumentParser(description="CPU realtime rPPG demo using POS/CHROM.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--method", choices=["pos", "chrom"], default="pos")
    parser.add_argument("--window", type=float, default=12.0, help="Signal window in seconds.")
    parser.add_argument("--width", type=int, default=640, help="Camera capture width.")
    parser.add_argument("--height", type=int, default=480, help="Camera capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS.")
    parser.add_argument("--detector", choices=["yunet", "haar"], default="yunet", help="Face detector backend.")
    parser.add_argument("--yunet-model", default=str(DEFAULT_YUNET_MODEL), help="Path to YuNet ONNX model.")
    parser.add_argument("--face-confidence", type=float, default=0.90, help="Minimum YuNet face confidence.")
    return parser.parse_args()


def bandpass_fft(signal, fs, low_hz=HR_MIN_HZ, high_hz=HR_MAX_HZ):
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)
    if len(signal) < 8:
        return signal

    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    spec = np.fft.rfft(signal * np.hanning(len(signal)))
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spec[~mask] = 0
    return np.fft.irfft(spec, n=len(signal))


def uniform_resample(times, values, target_fs=TARGET_FS):
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(times) < 2:
        return values, target_fs

    duration = times[-1] - times[0]
    if duration <= 0:
        return values, target_fs

    n = max(8, int(duration * target_fs))
    grid = np.linspace(times[0], times[-1], n)
    out = np.empty((n, values.shape[1]), dtype=np.float64)
    for i in range(values.shape[1]):
        out[:, i] = np.interp(grid, times, values[:, i])
    return out, target_fs


def normalize_rgb(rgb):
    c = np.asarray(rgb, dtype=np.float64)
    return c / (np.mean(c, axis=0, keepdims=True) + 1e-8) - 1.0


def pos_signal(rgb, fs, win_seconds=POS_WINDOW_SECONDS, step_seconds=POS_STEP_SECONDS):
    rgb = np.asarray(rgb, dtype=np.float64)
    n = len(rgb)
    win = max(8, int(round(win_seconds * fs)))
    if n < win:
        return pos_signal_segment(rgb)
    step = max(1, int(round(step_seconds * fs)))

    pulse = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)

    for start in range(0, n - win + 1, step):
        stop = start + win
        segment = pos_signal_segment(rgb[start:stop])
        segment = segment - np.mean(segment)
        pulse[start:stop] += segment
        counts[start:stop] += 1.0

    if counts[-1] == 0:
        start = n - win
        stop = n
        segment = pos_signal_segment(rgb[start:stop])
        segment = segment - np.mean(segment)
        pulse[start:stop] += segment
        counts[start:stop] += 1.0

    valid = counts > 0
    pulse[valid] /= counts[valid]
    return pulse


def pos_signal_segment(rgb):
    c = normalize_rgb(rgb).T
    h = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]]) @ c
    std0 = np.std(h[0]) + 1e-8
    std1 = np.std(h[1]) + 1e-8
    return h[0] + (std0 / std1) * h[1]


def chrom_signal(rgb, fs, win_seconds=CHROM_WINDOW_SECONDS, step_seconds=CHROM_STEP_SECONDS):
    rgb = np.asarray(rgb, dtype=np.float64)
    n = len(rgb)
    win = max(8, int(round(win_seconds * fs)))
    if n < win:
        return chrom_signal_segment(rgb, fs)
    step = max(1, int(round(step_seconds * fs)))

    pulse = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)

    for start in range(0, n - win + 1, step):
        stop = start + win
        segment = chrom_signal_segment(rgb[start:stop], fs)
        segment = segment - np.mean(segment)
        pulse[start:stop] += segment
        counts[start:stop] += 1.0

    if counts[-1] == 0:
        start = n - win
        stop = n
        segment = chrom_signal_segment(rgb[start:stop], fs)
        segment = segment - np.mean(segment)
        pulse[start:stop] += segment
        counts[start:stop] += 1.0

    valid = counts > 0
    pulse[valid] /= counts[valid]
    return pulse


def chrom_signal_segment(rgb, fs):
    c = normalize_rgb(rgb)
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    x = 3.0 * r - 2.0 * g
    y = 1.5 * r + g - 1.5 * b
    x_filtered = bandpass_fft(x, fs, HR_MIN_HZ, HR_MAX_HZ)
    y_filtered = bandpass_fft(y, fs, HR_MIN_HZ, HR_MAX_HZ)
    alpha = (np.std(x_filtered) + 1e-8) / (np.std(y_filtered) + 1e-8)
    return x_filtered - alpha * y_filtered


def estimate_hr_from_samples(samples, fs, method):
    if len(samples) < MIN_ESTIMATE_SAMPLES:
        return None, None, None, None, 0.0

    raw = pos_signal(samples, fs) if method == "pos" else chrom_signal(samples, fs)
    filtered = bandpass_fft(raw, fs, HR_MIN_HZ, HR_MAX_HZ)

    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fs)
    spec = np.abs(np.fft.rfft(filtered * np.hanning(len(filtered)))) ** 2
    mask = (freqs >= HR_MIN_HZ) & (freqs <= HR_MAX_HZ)
    if not np.any(mask):
        return filtered, freqs, spec, None, 0.0

    band_freqs = freqs[mask]
    band_spec = spec[mask]
    peak_idx = int(np.argmax(band_spec))
    bpm = float(band_freqs[peak_idx] * 60.0)

    peak_freq = band_freqs[peak_idx]
    peak_mask = np.abs(band_freqs - peak_freq) <= 0.10
    peak_power = float(np.sum(band_spec[peak_mask]))
    noise_power = float(np.sum(band_spec[~peak_mask]) + 1e-8)
    confidence = np.clip(peak_power / noise_power, 0.0, 5.0) / 5.0
    return filtered, freqs, spec, bpm, float(confidence)


def estimate_hr(times, rgb, method):
    if len(rgb) < MIN_ESTIMATE_SAMPLES:
        return None, None, None, None, 0.0

    samples, fs = uniform_resample(times, rgb, target_fs=TARGET_FS)
    if len(samples) < MIN_ESTIMATE_SAMPLES:
        return None, None, None, None, 0.0

    return estimate_hr_from_samples(samples, fs, method)


def load_haar_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")
    return {"kind": "haar", "model": detector}


def load_yunet_detector(model_path, frame_size, score_threshold):
    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError("This OpenCV build does not include cv2.FaceDetectorYN.")
    if not Path(model_path).exists():
        raise RuntimeError(f"YuNet model not found: {model_path}")
    detector = cv2.FaceDetectorYN.create(
        str(model_path),
        "",
        frame_size,
        score_threshold,
        0.3,
        5000,
    )
    return {"kind": "yunet", "model": detector, "frame_size": frame_size}


def load_face_detector(kind, yunet_model, frame_size, face_confidence):
    if kind == "haar":
        return load_haar_detector()

    try:
        return load_yunet_detector(yunet_model, frame_size, face_confidence)
    except RuntimeError as exc:
        print(f"YuNet unavailable ({exc}); falling back to Haar.")
        return load_haar_detector()


def choose_face(faces, frame_shape):
    if len(faces) == 0:
        return None

    h, w = frame_shape[:2]
    center = np.array([w / 2.0, h / 2.0])

    def score(face):
        x, y, fw, fh = face
        face_center = np.array([x + fw / 2.0, y + fh / 2.0])
        return fw * fh - 0.25 * np.linalg.norm(face_center - center)

    return tuple(max(faces, key=score))


def clamp_face_box(face, frame_shape):
    x, y, w, h = face
    frame_h, frame_w = frame_shape[:2]
    x1 = max(0, min(frame_w - 1, x))
    y1 = max(0, min(frame_h - 1, y))
    x2 = max(0, min(frame_w, x + w))
    y2 = max(0, min(frame_h, y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def detect_face_yunet(frame, detector_state):
    frame_h, frame_w = frame.shape[:2]
    frame_size = (frame_w, frame_h)
    if detector_state.get("frame_size") != frame_size:
        detector_state["model"].setInputSize(frame_size)
        detector_state["frame_size"] = frame_size

    _, faces = detector_state["model"].detect(frame)
    if faces is None or len(faces) == 0:
        return None

    boxes = []
    for face in faces:
        x, y, w, h = face[:4]
        box = clamp_face_box((int(round(x)), int(round(y)), int(round(w)), int(round(h))), frame.shape)
        if box is not None:
            boxes.append(box)
    return choose_face(boxes, frame.shape)


def detect_face_haar(frame, detector_state):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector_state["model"].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90))
    return choose_face(faces, gray.shape)


def detect_face(frame, detector_state):
    if detector_state["kind"] == "yunet":
        try:
            return detect_face_yunet(frame, detector_state)
        except cv2.error as exc:
            print(f"YuNet detect failed ({exc}); switching to Haar.")
            detector_state.clear()
            detector_state.update(load_haar_detector())
            return detect_face_haar(frame, detector_state)
    return detect_face_haar(frame, detector_state)


def smooth_face_box(previous, detected, alpha=0.25):
    if detected is None:
        return previous
    if previous is None:
        return tuple(int(v) for v in detected)

    px, py, pw, ph = previous
    dx, dy, dw, dh = detected
    smoothed = (
        alpha * dx + (1.0 - alpha) * px,
        alpha * dy + (1.0 - alpha) * py,
        alpha * dw + (1.0 - alpha) * pw,
        alpha * dh + (1.0 - alpha) * ph,
    )
    return tuple(int(round(v)) for v in smoothed)


def roi_mask_for_face(frame_shape, face):
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


def mean_rgb(frame_bgr, mask):
    pixels = frame_bgr[mask > 0]
    if len(pixels) < 64:
        return None
    rgb = pixels[:, ::-1]
    return np.mean(rgb, axis=0)


def empty_display_results():
    return {
        method: {
            "bpm": None,
            "confidence": None,
            "last_reliable_time": 0.0,
            "raw_bpm": None,
            "raw_confidence": None,
        }
        for method in RPPG_METHODS
    }


def empty_signal_results():
    return {method: {"signal": None, "freqs": None, "spec": None} for method in RPPG_METHODS}


def clear_signal_state(rgb_history, time_history):
    rgb_history.clear()
    time_history.clear()
    return empty_display_results(), empty_signal_results()


def update_display_result(display_results, method, bpm, confidence, now):
    result = display_results[method]
    result["raw_bpm"] = bpm
    result["raw_confidence"] = confidence
    if bpm is not None and confidence >= MIN_DISPLAY_CONFIDENCE:
        result["bpm"] = bpm
        result["confidence"] = confidence
        result["last_reliable_time"] = now
    elif now - result["last_reliable_time"] > DISPLAY_HOLD_SECONDS:
        result["bpm"] = None
        result["confidence"] = None


def draw_plot(canvas, x, y, w, h, values, color, label, vline=None):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (45, 45, 45), 1)
    cv2.putText(canvas, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)
    if values is None or len(values) < 2:
        return

    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return
    vals = vals[-w:]
    vmin, vmax = np.percentile(vals, [5, 95])
    if abs(vmax - vmin) < 1e-8:
        return
    ys = y + h - np.clip((vals - vmin) / (vmax - vmin), 0, 1) * h
    xs = np.linspace(x, x + w - 1, len(vals))
    pts = np.column_stack([xs, ys]).astype(np.int32)
    cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
    if vline is not None:
        vx = int(x + np.clip(vline, 0, 1) * w)
        cv2.line(canvas, (vx, y), (vx, y + h), (150, 150, 150), 1)


def draw_bpm_row(canvas, x, y, label, result, color):
    bpm = result["bpm"]
    confidence = result["confidence"]
    bpm_text = "--" if bpm is None else f"{bpm:5.1f}"
    conf = 0.0 if confidence is None else confidence
    conf_text = "--" if confidence is None else f"{confidence:0.2f}"

    cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 1)
    cv2.putText(canvas, bpm_text, (x + 92, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.35, color, 3)
    cv2.putText(canvas, "BPM", (x + 245, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1)
    cv2.putText(
        canvas,
        f"Conf: {conf_text}",
        (x, y + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (185, 190, 198),
        1,
    )
    cv2.rectangle(canvas, (x + 92, y + 25), (x + 280, y + 38), (70, 70, 70), 1)
    cv2.rectangle(canvas, (x + 92, y + 25), (x + 92 + int(188 * conf), y + 38), color, -1)


def draw_raw_result(canvas, x, y, label, result):
    raw_bpm = result["raw_bpm"]
    raw_confidence = result["raw_confidence"]
    bpm_text = "--" if raw_bpm is None else f"{raw_bpm:5.1f}"
    conf_text = "--" if raw_confidence is None else f"{raw_confidence:0.2f}"
    cv2.putText(
        canvas,
        f"raw {label}: {bpm_text} BPM  conf {conf_text}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (150, 155, 164),
        1,
    )


def draw_dashboard(
    frame,
    face,
    roi_rects,
    method,
    display_results,
    selected_signal,
    fps,
    window_seconds,
):
    frame_h, frame_w = frame.shape[:2]
    panel_w = 420
    canvas_h = max(frame_h, 720)
    canvas = np.zeros((canvas_h, frame_w + panel_w, 3), dtype=np.uint8)
    canvas[:frame_h, :frame_w] = frame
    panel_x = frame_w
    canvas[:, panel_x:] = (24, 26, 29)

    if face is not None:
        x, y, w, h = face
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (60, 190, 120), 2)
    if roi_rects:
        for x, y, w, h in roi_rects:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (90, 210, 250), 1)

    px = panel_x + 24
    cv2.putText(canvas, "CPU rPPG Demo", (px, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (245, 245, 245), 2)
    info_lines = [
        f"Method: {method.upper()}",
        f"Camera FPS: {fps:4.1f}",
        f"Window: {window_seconds:4.1f}s",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(canvas, line, (px, 78 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 205, 220), 1)

    cv2.putText(
        canvas,
        f"BPM confidence min {MIN_DISPLAY_CONFIDENCE:0.2f}",
        (px, 156),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (165, 170, 178),
        1,
    )
    draw_bpm_row(canvas, px, 192, "POS", display_results["pos"], (80, 220, 140))
    draw_bpm_row(canvas, px, 264, "CHROM", display_results["chrom"], (90, 210, 250))
    draw_raw_result(canvas, px, 330, "CHROM", display_results["chrom"])

    plot_w = panel_w - 48
    plot_h = 82
    wave_y = 385
    spec_y = wave_y + plot_h + 58
    help_y = spec_y + plot_h + 44
    signal = selected_signal["signal"]
    freqs = selected_signal["freqs"]
    spec = selected_signal["spec"]
    selected_bpm = display_results[method]["bpm"]
    draw_plot(canvas, px, wave_y, plot_w, plot_h, signal, (90, 210, 250), f"{method.upper()} waveform")

    spectrum_values = None
    peak_line = None
    if freqs is not None and spec is not None:
        mask = (freqs >= HR_MIN_HZ) & (freqs <= HR_MAX_HZ)
        if np.any(mask):
            spectrum_values = spec[mask]
            if selected_bpm is not None:
                peak_hz = selected_bpm / 60.0
                peak_line = (peak_hz - HR_MIN_HZ) / (HR_MAX_HZ - HR_MIN_HZ)
    draw_plot(canvas, px, spec_y, plot_w, plot_h, spectrum_values, (120, 170, 255), "spectrum", peak_line)

    help_lines = [
        "q quit    r reset",
        "p/c select plot",
        "+/- window size",
        "Keep face still and well lit.",
    ]
    for i, line in enumerate(help_lines):
        cv2.putText(canvas, line, (px, help_y + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (165, 170, 178), 1)

    return canvas


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    detector = load_face_detector(
        args.detector,
        args.yunet_model,
        (args.width, args.height),
        args.face_confidence,
    )

    method = args.method
    window_seconds = args.window
    rgb_history = deque()
    time_history = deque()
    previous_face = None
    displayed_face = None
    last_detection = 0.0
    last_face_seen_time = 0.0
    last_t = time.perf_counter()
    fps_smooth = 0.0
    display_results = empty_display_results()
    signal_results = empty_signal_results()
    last_estimate = 0.0

    cv2.namedWindow("CPU rPPG Demo", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.perf_counter()
        dt = max(1e-6, now - last_t)
        last_t = now
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt) if fps_smooth else 1.0 / dt

        frame = cv2.flip(frame, 1)
        if now - last_detection > DETECTION_INTERVAL_S:
            detected_face = detect_face(frame, detector)
            if detected_face is not None:
                previous_face = detected_face
                last_face_seen_time = now
            last_detection = now

        if previous_face is not None and now - last_face_seen_time > FACE_LOST_CLEAR_SECONDS:
            previous_face = None
            displayed_face = None
            display_results, signal_results = clear_signal_state(rgb_history, time_history)
            last_estimate = now

        displayed_face = smooth_face_box(displayed_face, previous_face)

        mask, roi_rects = roi_mask_for_face(frame.shape, displayed_face)
        rgb = mean_rgb(frame, mask)
        if rgb is not None:
            rgb_history.append(rgb)
            time_history.append(now)

        while time_history and now - time_history[0] > window_seconds:
            time_history.popleft()
            rgb_history.popleft()

        if len(rgb_history) >= MIN_ESTIMATE_SAMPLES and now - last_estimate >= ESTIMATE_INTERVAL_S:
            times = np.array(time_history)
            rgbs = np.array(rgb_history)
            samples, fs = uniform_resample(times, rgbs, target_fs=TARGET_FS)
            for rppg_method in RPPG_METHODS:
                signal, freqs, spec, bpm, confidence = estimate_hr_from_samples(samples, fs, rppg_method)
                signal_results[rppg_method] = {"signal": signal, "freqs": freqs, "spec": spec}
                update_display_result(display_results, rppg_method, bpm, confidence, now)
            last_estimate = now

        dashboard = draw_dashboard(
            frame,
            displayed_face,
            roi_rects,
            method,
            display_results,
            signal_results[method],
            fps_smooth,
            window_seconds,
        )
        cv2.imshow("CPU rPPG Demo", dashboard)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            method = "pos"
        elif key == ord("c"):
            method = "chrom"
        elif key == ord("r"):
            display_results, signal_results = clear_signal_state(rgb_history, time_history)
        elif key in (ord("+"), ord("=")):
            window_seconds = min(30.0, window_seconds + 1.0)
        elif key in (ord("-"), ord("_")):
            window_seconds = max(6.0, window_seconds - 1.0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
