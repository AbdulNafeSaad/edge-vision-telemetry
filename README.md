# Edge-Deployed Computer Vision & Telemetry Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ultralytics YOLOv9](https://img.shields.io/badge/YOLOv9-RealTime-brightgreen.svg)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green.svg)](https://opencv.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-lightgrey.svg)](https://onnxruntime.ai/)
[![Intel OpenVINO](https://img.shields.io/badge/Intel-OpenVINO-0071C5.svg)](https://docs.openvino.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An optimized computer vision inference and hardware telemetry pipeline engineered for resource-constrained edge environments. The system compiles YOLOv9 PyTorch checkpoints into graph-simplified **ONNX** and **Intel OpenVINO IR** runtimes while capturing real-time bounding box detections, live performance overlays, and per-frame latency, throughput, CPU, and memory telemetry.

---

# Key Features

- **Multi-Runtime Graph Optimization (`export.py`)**: Converts `.pt` checkpoints into static-shape ONNX and OpenVINO Intermediate Representation (`model.xml` + `model.bin`) artifacts and audits storage footprint deltas.
- **Live Performance HUD & Visual Analytics**: Renders real-time OpenCV bounding boxes, class labels, and confidence scores alongside an on-screen telemetry HUD (`FPS`, `Latency ms`, `CPU %`, `RAM MB`, and active object count).
- **Frame-Level CSV Telemetry (`logs/`)**: Streams structured per-frame runtime metrics (`inference_ms`, `fps`, `cpu_percent`, `ram_mb`, `detections_count`, `detected_classes`, `mean_confidence`) via `psutil` for downstream performance auditing.
- **CLI-Configurable & Headless Ready**: Supports live webcams, video files, or RTSP streams with customizable confidence thresholds (`--conf`), frame limits (`--max-frames`), and headless benchmarking (`--no-display`).

---

# Execution Architecture

```text
[ YOLOv9 .pt Weights ] ──► [ export.py ] ──► [ ONNX / OpenVINO IR Artifacts ]
                                                            │
                                                            ▼
[ Camera / Video Stream ] ──► [ OpenCV Capture ] ──► [ Edge Inference Engine ]
                                                            │
                                          ┌─────────────────┴─────────────────┐
                                          ▼                                   ▼
                              [ Annotated HUD Window ]            [ Frame-Level CSV Logs ]
```

---

# Empirical Benchmark Results (200 Frames @ 640×480)

Evaluated on an **Intel Core i7-3610QM CPU @ 2.30GHz** (first 10 warmup frames excluded from steady-state calculations to isolate Frame 1 graph compilation overhead):

| Runtime Format | Artifact | Disk Size | Cold-Start (Frame 1) | Steady Mean Latency | Steady P95 Latency | Avg Throughput | Peak RAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch Baseline** | `yolov9t.pt` | 4.74 MB | 4517.93 ms | 319.41 ms | 418.61 ms | 3.12 FPS | 402.4 MB |
| **ONNX Runtime** | `yolov9t.onnx` | 8.38 MB | 4734.66 ms | 206.02 ms | **251.14 ms** | 4.81 FPS (+54.2%) | 454.9 MB |
| **Intel OpenVINO IR** | `yolov9t_openvino_model/` | 8.72 MB | 4998.59 ms | **198.75 ms** | 268.34 ms | **5.03 FPS (+61.2%)** | 456.2 MB |

### Key Engineering Takeaways
- **Throughput vs. Memory Trade-off**: Intel OpenVINO IR cuts steady-state CPU latency by **37.8%** (`319.41 ms` → `198.75 ms`) and boosts throughput by **+61.2%**, requiring ~54 MB of additional runtime memory for compiled execution graphs.
- **Tail-Latency Consistency**: ONNX Runtime (`CPUExecutionProvider`) achieved the lowest P95 tail latency (**251.14 ms** vs. `418.61 ms` in PyTorch), providing predictable frame deadlines on CPU-bound hardware.
- **Cold-Start Overhead**: Initial graph compilation adds ~217–481 ms on Frame 1 in ONNX/OpenVINO compared to eager PyTorch execution before stabilizing across subsequent frames.

---

# Tech Stack

- **Computer Vision & Video I/O**: OpenCV (`opencv-python`)
- **Model Engine**: Ultralytics YOLO (YOLOv9)
- **Edge Inference Runtimes**: ONNX Runtime (`CPUExecutionProvider`), Intel OpenVINO
- **System Telemetry & Profiling**: `psutil`, NumPy
- **Language**: Python 3.10+

---

# Repository Structure

```text
├── export.py                                                 # PyTorch-to-ONNX & OpenVINO IR export utility
├── edge_inference.py                                         # CLI inference loop, HUD renderer & telemetry logger
├── requirements.txt                                          # Pinned project dependencies
├── logs/
│   └── telemetry_yolov9t_openvino_model_20261120_214351.csv  # Sample 200-frame hardware & detection telemetry log
└── README.md
```

---

# Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Export Optimized Edge Models
Compile `yolov9t.pt` into graph-simplified ONNX and OpenVINO IR formats:
```bash
python export.py --weights yolov9t.pt --formats onnx openvino
```

### 3. Run Real-Time Edge Inference
Launch live webcam detection (`--source 0`) using the OpenVINO runtime:
```bash
python edge_inference.py --model yolov9t_openvino_model --source 0 --conf 0.45
```

To run a headless 200-frame benchmark without opening the GUI window:
```bash
python edge_inference.py --model yolov9t_openvino_model --source 0 --max-frames 200 --no-display
```

*Note: Press `q` or close the display window (`X`) at any time to cleanly terminate the stream and print the session benchmark summary.*

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
