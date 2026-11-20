import argparse
from pathlib import Path
from ultralytics import YOLO


def get_size_mb(path: Path) -> float:
    """Return file or directory size in megabytes."""
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv9 models for edge deployment.")
    parser.add_argument("--weights", type=str, default="yolov9t.pt", help="Source .pt weights (e.g., yolov9t.pt, yolov9c.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Static input image size for export")
    parser.add_argument("--formats", nargs="+", default=["onnx", "openvino"], choices=["onnx", "openvino"], help="Target export formats")
    args = parser.parse_args()

    model = YOLO(args.weights)
    base_path = Path(args.weights)
    base_size = get_size_mb(base_path)

    results_summary = [("PyTorch (.pt)", str(base_path), base_size, "Baseline")]

    for fmt in args.formats:
        print(f"\n[INFO] Exporting {args.weights} to {fmt.upper()} (imgsz={args.imgsz})...")
        exported_path = Path(model.export(format=fmt, imgsz=args.imgsz, simplify=True))
        exp_size = get_size_mb(exported_path)
        delta_pct = ((exp_size - base_size) / base_size) * 100 if base_size > 0 else 0.0
        results_summary.append((fmt.upper(), str(exported_path), exp_size, f"{delta_pct:+.1f}%"))

    print("\n" + "=" * 72)
    print(f"{'Format':<16} | {'Artifact Path':<30} | {'Size (MB)':<10} | {'Delta'}")
    print("-" * 72)
    for fmt_name, artifact, size_mb, delta in results_summary:
        print(f"{fmt_name:<16} | {artifact:<30} | {size_mb:<10.2f} | {delta}")
    print("=" * 72)


if __name__ == "__main__":
    main()