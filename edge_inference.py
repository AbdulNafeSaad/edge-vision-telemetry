import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import psutil
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Edge Computer Vision Inference & Telemetry Engine")
    parser.add_argument("--model", type=str, default="yolov9t.onnx", help="Path to .pt, .onnx, or _openvino_model/ directory")
    parser.add_argument("--source", type=str, default="0", help="Camera index (e.g., 0) or path to video file")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop automatically after N frames (0 = unlimited)")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to store CSV telemetry logs")
    parser.add_argument("--no-display", action="store_true", help="Run headless without OpenCV GUI window")
    return parser.parse_args()


def draw_hud(frame, fps: float, infer_ms: float, cpu_pct: float, ram_mb: float, det_count: int, model_label: str):
    """Render a translucent telemetry HUD in the top-left corner."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (315, 145), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    lines = [
        f"Model:   {model_label}",
        f"FPS:     {fps:.1f}  |  Latency: {infer_ms:.1f} ms",
        f"CPU:     {cpu_pct:.1f}% |  RAM: {ram_mb:.0f} MB",
        f"Objects: {det_count}",
    ]
    for idx, text in enumerate(lines):
        y = 35 + idx * 28
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 140), 2, cv2.LINE_AA)


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    model_path = Path(args.model)
    model_label = model_path.name
    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model, task="detect")

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_path.stem.replace(" ", "_")
    csv_path = Path(args.log_dir) / f"telemetry_{safe_model_name}_{timestamp_tag}.csv"

    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)  # Prime psutil CPU counter

    window_name = "Edge Vision System"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    latencies_ms = []
    fps_history = []
    ram_history = []
    frame_id = 0
    ema_fps = 0.0

    print(f"[INFO] Logging frame telemetry to: {csv_path}")
    print("[INFO] Press 'q' or close the window to stop and view benchmark summary.")

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_id",
            "timestamp",
            "model",
            "inference_ms",
            "fps",
            "cpu_percent",
            "ram_mb",
            "detections_count",
            "detected_classes",
            "mean_confidence",
        ])

        while cap.isOpened():
            loop_start = time.perf_counter()
            success, frame = cap.read()
            if not success:
                break

            frame_id += 1

            # Timed inference pass
            t0 = time.perf_counter()
            results = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            # Parse detections
            boxes = results[0].boxes
            det_count = len(boxes) if boxes is not None else 0
            if det_count > 0:
                cls_ids = boxes.cls.int().tolist()
                confs = boxes.conf.tolist()
                names_map = model.names if isinstance(model.names, dict) else {}
                classes_str = "|".join(names_map.get(c, str(c)) for c in cls_ids)
                mean_conf = round(sum(confs) / det_count, 4)
            else:
                classes_str = "none"
                mean_conf = 0.0

            # System resource metrics
            cpu_pct = process.cpu_percent(interval=None)
            ram_mb = process.memory_info().rss / (1024 * 1024)

            loop_elapsed = max(time.perf_counter() - loop_start, 1e-6)
            instant_fps = 1.0 / loop_elapsed
            ema_fps = instant_fps if frame_id == 1 else (0.9 * ema_fps + 0.1 * instant_fps)

            latencies_ms.append(infer_ms)
            fps_history.append(ema_fps)
            ram_history.append(ram_mb)

            writer.writerow([
                frame_id,
                datetime.now().isoformat(timespec="milliseconds"),
                model_label,
                round(infer_ms, 2),
                round(ema_fps, 2),
                round(cpu_pct, 1),
                round(ram_mb, 1),
                det_count,
                classes_str,
                mean_conf,
            ])

            if not args.no_display:
                annotated_frame = results[0].plot()
                draw_hud(annotated_frame, ema_fps, infer_ms, cpu_pct, ram_mb, det_count, model_label)
                cv2.imshow(window_name, annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

            if args.max_frames > 0 and frame_id >= args.max_frames:
                break

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    if len(latencies_ms) > 10:
        cold_start_ms = latencies_ms[0]
        steady_lat = np.array(latencies_ms[10:])
        steady_fps = np.array(fps_history[10:])
        print("\n" + "=" * 58)
        print(f"BENCHMARK SUMMARY — {model_label} ({frame_id} total frames)")
        print("=" * 58)
        print(f"Cold-Start (Frame 1)   : {cold_start_ms:.2f} ms")
        print(f"Steady Mean Latency    : {steady_lat.mean():.2f} ms")
        print(f"Steady P95 Latency     : {np.percentile(steady_lat, 95):.2f} ms")
        print(f"Steady Throughput      : {steady_fps.mean():.2f} FPS")
        print(f"Peak Process RAM       : {max(ram_history):.1f} MB")
        print(f"Telemetry Log Saved    : {csv_path}")
        print("=" * 58)

if __name__ == "__main__":
    main()