# CamHeartBeat：摄像头实时心率检测 Demo

这是一个在本地 Mac 上运行的 CPU 实时心率估计 Demo。程序使用普通摄像头采集人脸画面，通过传统 rPPG 算法 `POS` / `CHROM` 从面部颜色变化中估算心率。

不需要 NVIDIA GPU，也不需要深度学习心率模型。

## 功能

- 本地摄像头实时输入
- CPU 实时运行
- 默认使用 YuNet 人脸检测，Haar 作为备用方案
- 支持 `POS` 和 `CHROM` 两种传统 rPPG 算法
- 显示实时面板：BPM、置信度、rPPG 波形、频谱、FPS
- 低置信度结果过滤：BPM confidence 低于 `0.20` 时不会更新显示结果

## 原理

心脏搏动会让面部皮肤的血液含量发生微小周期变化，进而造成 RGB 颜色信号的轻微波动。程序大致流程是：

1. 用 YuNet 或 Haar 检测人脸。
2. 在额头和两侧脸颊选取 ROI。
3. 计算 ROI 内平均 RGB 值，形成随时间变化的颜色序列。
4. 用 `POS` 或 `CHROM` 把 RGB 序列转换为 rPPG 脉搏波信号。
5. 在 `45-180 BPM` 范围内做频谱分析，寻找主峰并换算为心率。
6. 根据主峰能量和噪声能量估计 confidence score。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

依赖见 [requirements.txt](requirements.txt)。

当前代码已在 `opencv-python 4.6.0` 环境下兼容 YuNet `2022mar` 模型。较新的 OpenCV 版本通常也可以运行。

## 运行

默认运行：

```bash
python realtime_rppg_demo.py
```

切换到 CHROM：

```bash
python realtime_rppg_demo.py --method chrom
```

调整信号窗口：

```bash
python realtime_rppg_demo.py --window 15
```

指定摄像头：

```bash
python realtime_rppg_demo.py --camera 0
```

切回 Haar 人脸检测：

```bash
python realtime_rppg_demo.py --detector haar
```

调整 YuNet 人脸检测阈值：

```bash
python realtime_rppg_demo.py --detector yunet --face-confidence 0.90
```

## 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--camera` | `0` | OpenCV 摄像头索引 |
| `--method` | `pos` | rPPG 算法，可选 `pos` / `chrom` |
| `--window` | `12.0` | 心率估计使用的滑动窗口长度，单位秒 |
| `--width` | `640` | 请求摄像头采集宽度 |
| `--height` | `480` | 请求摄像头采集高度 |
| `--fps` | `30` | 请求摄像头帧率 |
| `--detector` | `yunet` | 人脸检测器，可选 `yunet` / `haar` |
| `--yunet-model` | 内置模型 | YuNet ONNX 模型路径 |
| `--face-confidence` | `0.90` | YuNet 最小人脸检测置信度 |

内置 YuNet 模型路径：

```text
models/face_detection_yunet_2022mar.onnx
```

如果 YuNet 初始化或运行失败，程序会自动回退到 Haar 检测器，避免 demo 直接崩溃。

## 面板说明

右侧面板会显示：

- `Method`：当前算法，`POS` 或 `CHROM`
- `Camera FPS`：当前摄像头处理帧率
- `Window`：当前信号窗口长度
- `BPM`：当前显示的心率
- `Confidence`：当前显示心率对应的置信度
- `rPPG waveform`：滤波后的 rPPG 波形
- `spectrum`：心率频段内的频谱

BPM confidence 阈值为 `0.20`：

- 新结果低于 `0.20` 时，不覆盖最近一次可信 BPM 和 confidence。
- 如果连续超过约 `5` 秒都没有可信结果，BPM 和 confidence 显示为 `--`。

## 快捷键

| 按键 | 功能 |
| --- | --- |
| `q` | 退出 |
| `p` | 切换到 POS |
| `c` | 切换到 CHROM |
| `r` | 重置信号缓存 |
| `+` / `-` | 增大 / 减小窗口长度 |

## 使用建议

- 保持光照明亮且稳定。
- 尽量避免强背光。
- 前 `10-15` 秒保持脸部稳定，等待信号窗口积累。
- 尽量不要大幅转头或快速移动。
- 眼镜反光、低光照、摄像头自动曝光变化都会降低置信度。
- macOS 首次运行时，需要允许 Terminal 或 Python 访问摄像头。

## 项目结构

```text
.
├── realtime_rppg_demo.py
├── requirements.txt
├── models/
│   ├── face_detection_yunet_2022mar.onnx
│   └── face_detection_yunet_2023mar.onnx
└── README.md
```

其中 `2022mar` 是当前默认模型，主要用于兼容 OpenCV 4.6。`2023mar` 保留在目录中，但当前默认不使用。

## 注意

这是研究和演示用途的程序，不是医疗设备。心率估计结果可能受光照、运动、肤色、摄像头质量、压缩和自动曝光影响，不能用于医疗诊断或健康决策。
# CamHeartBeat
