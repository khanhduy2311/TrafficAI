"""
Models Loader — Singleton YOLO model manager
Load 5 model một lần duy nhất, cache để tái sử dụng.
"""

from pathlib import Path
from ultralytics import YOLO
import threading

# Đường dẫn thư mục model (tương đối từ file này → ../../model/)
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "model"


class ModelsManager:
    """Singleton quản lý tất cả YOLO models."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._models = {}
        self._model_configs = {
            "vehicle_tracker": {
                "path": MODEL_DIR / "tracking.pt",
                "description": "Phát hiện & tracking phương tiện (Bus, Bike, Car, Pedestrian, Truck)",
                "classes": ["Bus", "Bike", "Car", "Pedestrian", "Truck"],
            },
            "traffic_light": {
                "path": MODEL_DIR / "detect_traffic_light.pt",
                "description": "Phát hiện trạng thái đèn giao thông (green, yellow, red, off)",
                "classes": ["green", "yellow", "red", "off"],
            },
            "lane_sign": {
                "path": MODEL_DIR / "detect_lane.pt",
                "description": "Phát hiện biển báo làn đường (R411, R412, R415)",
                "classes": ["R411", "R412", "R415"],
            },
            "helmet": {
                "path": MODEL_DIR / "detect_helmet.pt",
                "description": "Phát hiện đội mũ / không đội mũ (Bike, helmet, no helmet)",
                "classes": ["Bike", "helmet", "no helmet"],
            },
            "speed_sign": {
                "path": MODEL_DIR / "detect_speed_sign.pt",
                "description": "Phát hiện biển báo giới hạn tốc độ",
                "classes": ["100", "120", "20", "30", "40", "50", "60", "70", "80", "90"],
            },
        }

    def load_model(self, name: str) -> YOLO:
        """Load 1 model theo tên. Cache nếu đã load."""
        if name in self._models:
            return self._models[name]

        config = self._model_configs.get(name)
        if config is None:
            raise ValueError(f"Unknown model: {name}")

        model_path = config["path"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"[ModelsManager] Loading {name} from {model_path} ...")
        model = YOLO(str(model_path))
        self._models[name] = model
        print(f"[ModelsManager] ✓ {name} loaded — classes: {config['classes']}")
        return model

    def load_all(self):
        """Load tất cả 5 models."""
        for name in self._model_configs:
            try:
                self.load_model(name)
            except FileNotFoundError as e:
                print(f"[ModelsManager] ⚠ Skipping {name}: {e}")

    def get(self, name: str) -> YOLO | None:
        """Lấy model đã load (không load mới)."""
        return self._models.get(name)

    def is_loaded(self, name: str) -> bool:
        return name in self._models

    def loaded_models(self) -> list[str]:
        return list(self._models.keys())

    def all_configs(self) -> dict:
        result = {}
        for name, cfg in self._model_configs.items():
            result[name] = {
                "description": cfg["description"],
                "classes": cfg["classes"],
                "loaded": name in self._models,
                "path": str(cfg["path"]),
            }
        return result


# Global singleton
models_manager = ModelsManager()
