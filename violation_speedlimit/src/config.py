from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Model and tracker paths
VEHICLE_MODEL_PATH = str(PROJECT_ROOT / "models" / "yolo11m-2stg.pt")
SIGN_MODEL_PATH = str(PROJECT_ROOT / "models" / "speed_sign_detector.pt")
TRACKER_CONFIG = str(PROJECT_ROOT / "configs" / "my_tracker.yaml")

# Inference
IMG_SIZE = 960
CONF_THRESHOLD = 0.25

# Fisheye speed estimator
SENSOR_FOV_DEG = 185.0
REAL_RADIUS_METERS = 25.0
VALID_RADIUS_RATIO = 0.80
PPM_CENTER = 80.0

# Speed smoothing
SPEED_MEDIAN_WINDOW = 7
SPEED_STABLE_FRAMES = 5
MAX_JUMP_METERS_PER_SEC = 30.0

# Violation
DEFAULT_SPEED_LIMIT = 50
VIOLATION_THRESHOLD = 1.0
MIN_SPEED_TO_CHECK = 5

# Target classes
TARGET_CLASSES = {
    0: "Bus",
    1: "Bike",
    2: "Car",
    3: "Pedestrian",
    4: "Truck",
}

CLASS_COLORS = {
    "Bus": (255, 0, 0),
    "Bike": (0, 255, 255),
    "Car": (0, 255, 0),
    "Pedestrian": (0, 0, 255),
    "Truck": (255, 0, 255),
}

# Output
OUTPUT_VIDEO_PATH = str(PROJECT_ROOT / "output" / "result_violation.mp4")
VIOLATION_CSV_PATH = str(PROJECT_ROOT / "output" / "violation_report.csv")
EVIDENCE_DIR = str(PROJECT_ROOT / "violations")

# Speed sign detection
SIGN_DETECT_ONCE = True
SIGN_DETECT_NFRAMES = 3
SIGN_DETECT_CONF = 0.35
