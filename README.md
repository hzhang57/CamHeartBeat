# CPU rPPG Camera Heart Rate Demo

This is a local Mac-friendly real-time demo for estimating heart rate from a webcam using traditional rPPG algorithms: POS and CHROM.

It runs on CPU only. No NVIDIA GPU is required.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python realtime_rppg_demo.py
```

Optional:

```bash
python realtime_rppg_demo.py --method chrom --window 15 --camera 0
python realtime_rppg_demo.py --detector haar
python realtime_rppg_demo.py --detector yunet --face-confidence 0.90
```

YuNet is the default face detector. The bundled model is stored at:

```text
models/face_detection_yunet_2022mar.onnx
```

## Controls

- `q`: quit
- `p`: POS mode
- `c`: CHROM mode
- `r`: reset signal buffer
- `+` / `-`: increase or decrease the signal window

## Tips

- Use bright, stable lighting.
- Keep your face mostly still for the first 10-15 seconds.
- Avoid strong backlight and rapid head motion.
- On macOS, allow Terminal or your Python launcher to access the camera when prompted.

This is a research/demo implementation, not a medical device.
# CamHeartBeat
