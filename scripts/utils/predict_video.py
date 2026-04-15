import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2

def normalize(names):
    if names is None: return names
    if isinstance(names, dict):
        fixed = dict(names)
        if str(fixed.get(1)).lower() == "false": fixed[1] = "off"
        return fixed
    fixed = list(names)
    if len(fixed) > 1 and str(fixed[1]).lower() == "false": fixed[1] = "off"
    return fixed

def main():
    parser = argparse.ArgumentParser(description="YOLO MP4 Predictor")
    parser.add_argument("--weights", required=True, help="Path to weights")
    parser.add_argument("--source", required=True, help="Source video")
    parser.add_argument("--imgsz", type=int, default=960, help="Image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence")
    parser.add_argument("--output", default="outputs/predict.mp4", help="Output path")
    args = parser.parse_args()

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Dung codec mp4v pho bien
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print(f"🎬 Processing: {args.source} -> {args.output}")
    
    results = model.predict(source=args.source, imgsz=args.imgsz, conf=args.conf, stream=True, verbose=False)
    
    for i, r in enumerate(results):
        r.names = normalize(r.names)
        annotated = r.plot()
        writer.write(annotated)
        if i % 100 == 0:
            print(f"Processed {i} frames...")

    cap.release()
    writer.release()
    print(f"✅ Saved to: {args.output}")

if __name__ == "__main__":
    main()
