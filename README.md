# CamHeartBeat：摄像头实时心率检测 Demo

CamHeartBeat 是一个在本地 Mac 上运行的实时 rPPG 心率估计 demo 集合。仓库包含四个入口：

- `realtime_rppg_demo.py`：CPU 传统算法 demo，同时计算 `POS` 和 `CHROM`。
- `deep_rppg_demo.py`：基于 `open-rppg` 预训练深度模型的 demo。
- `bench_rppg_demo.py`：同一路摄像头输入下对比 `POS` 和深度模型估计结果。
- `bench_deep_rppg_demo.py`：同一路摄像头输入下对比两个 `open-rppg` 深度模型。

所有结果都只适合研究和演示，不是医疗读数。

## 功能

- 本地摄像头实时输入
- 传统 `POS` / `CHROM` rPPG 心率估计
- 可选 `open-rppg` 深度模型心率估计
- 同屏显示 BPM、质量分数、波形、频谱、FPS 和推理延迟
- 低质量结果过滤：低于阈值的新结果不会覆盖最近一次可信 BPM
- 人脸丢失后自动清空旧信号，避免继续用背景估计心率

## 原理

心脏搏动会让面部皮肤血液含量产生微小周期变化，进而造成 RGB 颜色信号轻微波动。传统 demo 的流程是：

1. 用 YuNet 或 Haar 检测人脸。
2. 在额头和两侧脸颊选取 ROI。
3. 计算 ROI 内平均 RGB，形成颜色时间序列。
4. 用 `POS` 和 `CHROM` 转换为 rPPG 脉搏波信号。两者都使用约 `1.6` 秒短窗 overlap-add，短窗步长约 `0.1` 秒。
5. 在 `45-180 BPM` 范围内做频谱分析，寻找主峰并换算为心率。
6. 根据主峰能量和噪声能量估计 confidence score。

深度 demo 和 bench demo 使用 `open-rppg` 的摄像头、人脸检测和预训练模型输出，并用 `SQI` 作为深度模型结果的质量分数。

## 安装

### 传统 POS/CHROM 环境

传统 demo 不需要深度学习依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` 当前锁定：

- `opencv-python==4.6.0.66`
- `numpy>=1.20`

YuNet 默认使用 `models/face_detection_yunet_2022mar.onnx`，用于兼容当前锁定的 OpenCV 4.6 环境。

### 深度模型 / 对比环境

`deep_rppg_demo.py`、`bench_rppg_demo.py` 和 `bench_deep_rppg_demo.py` 使用独立依赖：

```bash
python3 -m venv .venv-deep
source .venv-deep/bin/activate
pip install -r requirements-deep.txt
```

`requirements-deep.txt` 包含：

- `open-rppg`
- `setuptools<81`
- `opencv-python`
- `numpy`

`setuptools<81` 是为了兼容 `open-rppg` 当前仍会导入的旧模块 `pkg_resources`。

Apple Silicon Mac 上请确认 Python 是 ARM64 原生环境，不要在 Rosetta/x86 Python 里安装 JAX：

```bash
python -c "import platform; print(platform.machine())"
```

输出应为 `arm64`。如果输出 `x86_64`，建议使用 ARM64 版 Miniforge/Miniconda 后重新创建环境：

```bash
conda create -n rppg-arm python=3.10
conda activate rppg-arm
python -m pip install --upgrade pip
python -m pip install -r requirements-deep.txt
python -c "import platform; print(platform.machine())"
```

## 运行

### 传统 POS/CHROM Demo

```bash
python realtime_rppg_demo.py
```

常用命令：

```bash
python realtime_rppg_demo.py --method chrom
python realtime_rppg_demo.py --window 15
python realtime_rppg_demo.py --camera 0
python realtime_rppg_demo.py --detector haar
python realtime_rppg_demo.py --detector yunet --face-confidence 0.90
```

面板会同时显示 POS 和 CHROM 的 BPM 与 confidence。按 `p` / `c` 可以切换下方波形和频谱显示算法。

### 深度模型 Demo

```bash
python deep_rppg_demo.py
```

常用命令：

```bash
python deep_rppg_demo.py --camera 0 --model FacePhys.rlap --hr-window 10
python deep_rppg_demo.py --model EfficientPhys.rlap
python deep_rppg_demo.py --warmup 8 --min-sqi 0.20
```

面板会显示摄像头画面、人脸框、`Deep BPM`、`SQI`、模型名、FPS 和推理延迟。前几秒显示 `--` 是正常预热。

### POS / 深度模型对比 Demo

```bash
python bench_rppg_demo.py
```

常用命令：

```bash
python bench_rppg_demo.py --model FacePhys.rlap --window 12 --hr-window 10
python bench_rppg_demo.py --min-confidence 0.05 --min-sqi 0.20
python bench_rppg_demo.py --warmup 8 --hold 5
```

对比 demo 使用同一路 `open-rppg` 摄像头预览，同时计算传统 `POS` 和深度模型心率。面板会显示：

- POS BPM 和 confidence
- Deep Learning BPM 和 SQI
- 两者绝对差值
- Deep latency
- POS waveform 和 spectrum

如果 POS 有 `raw BPM` 但主显示仍是 `--`，通常是 confidence 低于 `--min-confidence`。当前对比 demo 默认阈值是 `0.05`。

### 深度模型对比 Demo

deep-vs-deep 对比 demo 只使用 `open-rppg` 已内置支持并随包携带权重的模型。默认使用 `FacePhys.rlap` 作为 baseline，`PhysFormer.rlap` 作为候选模型。`PhysFormer.rlap` 是 Transformer 系列深度 rPPG 模型，适合和默认稳健模型做效果对比。

```bash
python bench_deep_rppg_demo.py
```

常用命令：

```bash
python bench_deep_rppg_demo.py --candidate-model PhysFormer.rlap --baseline-model FacePhys.rlap
python bench_deep_rppg_demo.py --hr-window 10 --warmup 6 --min-sqi 0.20
```

面板会显示：

- FacePhys BPM、SQI、raw BPM、raw SQI、latency
- PhysFormer BPM、SQI、raw BPM、raw SQI、latency
- 两者绝对差值
- 摄像头 FPS、HR window 和 warmup

## open-rppg 模型推荐

当前只集成 `open-rppg` 模型，不接入外部框架或额外权重下载。

| 模型 | 建议用途 | 说明 |
| --- | --- | --- |
| `FacePhys.rlap` | 默认 baseline | 当前默认深度模型，实时 demo 已验证 |
| `PhysFormer.rlap` | 推荐候选 | Transformer 系列深度 rPPG 模型，当前 deep-vs-deep 默认候选 |
| `PhysNet.rlap` | 经典基线 | 经典 3D CNN rPPG 基线，适合和 FacePhys/PhysFormer 做补充对比 |
| `EfficientPhys.rlap` | 快速实验 | 较轻量，适合 Mac 上快速试跑 |
| `TSCAN.pure` | 快速实验 | 加载快，输入尺寸小 |
| `PhysFormer.*` | 重模型实验 | 更重，默认使用 `.rlap` 作为 deep-vs-deep 候选 |
| `PhysMamba.*` | 重模型实验 | 输入尺寸更大，启动和推理成本更高 |
| `RhythmMamba.*` | 重模型实验 | 输入尺寸更大，启动和推理成本更高 |

## 参数

### `realtime_rppg_demo.py`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--camera` | `0` | OpenCV 摄像头索引 |
| `--method` | `pos` | 下方波形和频谱显示算法，可选 `pos` / `chrom` |
| `--window` | `12.0` | 传统算法滑动窗口长度，单位秒 |
| `--width` | `640` | 请求摄像头采集宽度 |
| `--height` | `480` | 请求摄像头采集高度 |
| `--fps` | `30` | 请求摄像头帧率 |
| `--detector` | `yunet` | 人脸检测器，可选 `yunet` / `haar` |
| `--yunet-model` | 内置模型 | YuNet ONNX 模型路径 |
| `--face-confidence` | `0.90` | YuNet 最小人脸检测置信度 |

### `deep_rppg_demo.py`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--camera` | `0` | 摄像头索引 |
| `--model` | `FacePhys.rlap` | `open-rppg` 模型名 |
| `--hr-window` | `10.0` | 深度模型 HR 估计窗口，单位秒 |
| `--width` | `640` | 预览宽度 |
| `--height` | `480` | 预览高度 |
| `--min-sqi` | `0.20` | 更新显示 BPM 的最小 SQI |
| `--hold` | `5.0` | 保留最近可信 BPM 的秒数 |
| `--warmup` | `6.0` | 开始 HR 估计前收集信号秒数 |
| `--verbose-rppg` | 关闭 | 显示 `open-rppg` warning 日志 |

### `bench_rppg_demo.py`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--camera` | `0` | 摄像头索引 |
| `--model` | `FacePhys.rlap` | `open-rppg` 模型名 |
| `--window` | `12.0` | POS 滑动窗口长度，单位秒 |
| `--hr-window` | `10.0` | 深度模型 HR 估计窗口，单位秒 |
| `--width` | `640` | 预览宽度 |
| `--height` | `480` | 预览高度 |
| `--min-confidence` | `0.05` | 更新 POS BPM 的最小 confidence |
| `--min-sqi` | `0.20` | 更新深度模型 BPM 的最小 SQI |
| `--hold` | `5.0` | 保留最近可信 BPM 的秒数 |
| `--warmup` | `6.0` | 开始深度 HR 估计前收集信号秒数 |
| `--verbose-rppg` | 关闭 | 显示 `open-rppg` warning 日志 |

### `bench_deep_rppg_demo.py`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--camera` | `0` | 摄像头索引 |
| `--baseline-model` | `FacePhys.rlap` | baseline `open-rppg` 模型名 |
| `--candidate-model` | `PhysFormer.rlap` | candidate `open-rppg` 模型名 |
| `--hr-window` | `10.0` | HR 估计窗口，单位秒 |
| `--width` | `640` | 预览宽度 |
| `--height` | `480` | 预览高度 |
| `--min-sqi` | `0.20` | 更新 BPM 的最小 SQI |
| `--hold` | `5.0` | 保留最近可信 BPM 的秒数 |
| `--warmup` | `6.0` | 开始 HR 估计前收集信号秒数 |
| `--verbose-rppg` | 关闭 | 显示 `open-rppg` warning 日志和 HR 异常 |

## 快捷键

### 传统 Demo

| 按键 | 功能 |
| --- | --- |
| `q` | 退出 |
| `p` | 下方波形 / 频谱切换到 POS |
| `c` | 下方波形 / 频谱切换到 CHROM |
| `r` | 重置信号缓存 |
| `+` / `-` | 增大 / 减小窗口长度 |

### 深度 / 对比 Demo

| 按键 | 功能 |
| --- | --- |
| `q` | 退出 |

## 使用建议

- 保持光照明亮且稳定。
- 尽量避免强背光和大幅转头。
- 前 `10-15` 秒保持脸部稳定，等待信号窗口积累。
- 眼镜反光、低光照、摄像头自动曝光变化都会降低 confidence / SQI。
- macOS 首次运行时，需要允许 Terminal 或 Python 访问摄像头。
- macOS 上的 `AVFFrameReceiver`、`AVFAudioReceiver` 或 Continuity Camera warning 通常不影响 demo 运行。

## 项目结构

```text
.
├── VERSION
├── README.md
├── realtime_rppg_demo.py
├── deep_rppg_demo.py
├── bench_rppg_demo.py
├── bench_deep_rppg_demo.py
├── requirements.txt
├── requirements-deep.txt
└── models/
    ├── face_detection_yunet_2022mar.onnx
    └── face_detection_yunet_2023mar.onnx
```

其中 `2022mar` 是当前默认 YuNet 模型，主要用于兼容 OpenCV 4.6。`2023mar` 保留在目录中，但当前默认不使用。

## 版本

当前版本见 [VERSION](VERSION)。

## 注意

这是研究和演示用途的程序，不是医疗设备。心率估计结果可能受光照、运动、肤色、摄像头质量、压缩、自动曝光和模型适配影响，不能用于医疗诊断或健康决策。
