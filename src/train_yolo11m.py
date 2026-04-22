import os
from pathlib import Path

from ultralytics import YOLO


TRAIN_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = TRAIN_ROOT / "configs"


def resolve_data_config() -> str:
    for candidate in (
        CONFIG_DIR / "dataset_balanced.yaml",
        CONFIG_DIR / "dataset.yaml",
        CONFIG_DIR / "dataset.example.yaml",
    ):
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("No dataset config found in training/configs")


DATA_CONFIG = os.getenv("YOLO_DATA_CONFIG", resolve_data_config())
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "../m.pt")
TRAIN_DEVICE = os.getenv("YOLO_DEVICE")
PROJECT_DIR = os.getenv("YOLO_PROJECT_DIR", str(TRAIN_ROOT / "result_train_fisheye"))

model = YOLO(MODEL_PATH)

train_kwargs = {
    "data": DATA_CONFIG,
    "epochs": 100,
    "imgsz": 960,
    "batch": 16,
    "project": PROJECT_DIR,
    "name": "run_yolo11n",
    "patience": 30,
    "workers": 8,
    "mosaic": 1.0,
    "mixup": 0.1,
    "degrees": 10.0,
    "scale": 0.5,
    "fliplr": 0.0,
}

if TRAIN_DEVICE:
    train_kwargs["device"] = TRAIN_DEVICE

Path(PROJECT_DIR).mkdir(parents=True, exist_ok=True)
model.train(**train_kwargs)
