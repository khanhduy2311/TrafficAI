import argparse
import time
from pathlib import Path

import cv2


def normalize_class_names(names: dict | list | None) -> dict | list | None:
    """Override YAML-bool-parsed class names such as off -> False."""
    if names is None:
        return names

    if isinstance(names, dict):
        fixed = dict(names)
        if str(fixed.get(1)).lower() == "false":
            fixed[1] = "off"
        return fixed

    fixed = list(names)
    if len(fixed) > 1 and str(fixed[1]).lower() == "false":
        fixed[1] = "off"
    return fixed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO inference on a video and measure effective FPS."
    )
    parser.add_argument("--weights", required=True, help="Path to trained YOLO weights.")
    parser.add_argument("--source", required=True, help="Input video path.")
    parser.add_argument(
        "--output",
        default="outputs/predict/output.mp4",
        help="Output annotated video path.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference size. Lower is faster, higher is usually better for tiny objects.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold.")
    parser.add_argument("--device", default="0", help="CUDA device id or cpu.")
    return parser.parse_args()


def open_writer(output_path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")
    return writer


def main() -> None:
    args = parse_args()

    weights_path = Path(args.weights).resolve()
    source_path = Path(args.source).resolve()
    output_path = Path(args.output).resolve()

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Video not found: {source_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'ultralytics'. Install with: pip install -r requirements.txt"
        ) from exc

    model = YOLO(str(weights_path))
    if hasattr(model, "model") and hasattr(model.model, "names"):
        model.model.names = normalize_class_names(model.model.names)

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0:
        input_fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = open_writer(output_path, width, height, input_fps)

    frame_count = 0
    total_infer_time = 0.0
    total_loop_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        loop_start = time.perf_counter()
        infer_start = time.perf_counter()

        result = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]
        result.names = normalize_class_names(result.names)

        infer_elapsed = time.perf_counter() - infer_start
        total_infer_time += infer_elapsed

        annotated = result.plot()
        frame_count += 1

        avg_infer_fps = frame_count / total_infer_time if total_infer_time > 0 else 0.0

        loop_elapsed = time.perf_counter() - loop_start
        total_loop_time += loop_elapsed
        avg_end_to_end_fps = frame_count / total_loop_time if total_loop_time > 0 else 0.0

        cv2.putText(
            annotated,
            f"infer_fps: {avg_infer_fps:.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"end_to_end_fps: {avg_end_to_end_fps:.2f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(annotated)

    cap.release()
    writer.release()

    if frame_count == 0:
        raise RuntimeError("No frames were processed.")

    final_infer_fps = frame_count / total_infer_time if total_infer_time > 0 else 0.0
    final_end_to_end_fps = frame_count / total_loop_time if total_loop_time > 0 else 0.0

    print(f"Processed frames: {frame_count}")
    print(f"Average inference FPS: {final_infer_fps:.2f}")
    print(f"Average end-to-end FPS: {final_end_to_end_fps:.2f}")
    print(f"Annotated video saved to: {output_path}")


if __name__ == "__main__":
    main()
