# Traffic Speed Violation Detection

This module detects traffic speed violations from fisheye camera videos using YOLO and ByteTrack.

## Installation
```bash
cd fisheye_training
pip install -r requirements/inference.txt
```

## Basic Usage
```bash
python violation_speedlimit/src/main.py --source path/to/video.mp4
```

## Structure
```text
violation_speedlimit/
├─ configs/
│  └─ my_tracker.yaml
├─ models/
│  └─ .gitkeep
└─ src/
   ├─ config.py
   ├─ main.py
   ├─ pipeline.py
   ├─ core/
   └─ output/
      ├─ annotator.py
      ├─ evidence_saver.py
      └─ report_logger.py
```
