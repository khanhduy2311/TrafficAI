# TrafficAI - Wrong Lane Detection

Hệ thống phát hiện vi phạm chạy ngược chiều sử dụng YOLOv11 và tracking algorithm.

## Features

- 🚗 Phát hiện vehicle (Car, Bus, Truck, Bike, Pedestrian)
- 🎥 Tracking objects qua các frame
- 📊 Tính toán scene flow để phát hiện chuyển động ngược chiều
- 📹 Output video annotated với bounding boxes
- 📋 Export violations CSV

## Project Structure

`
.
├── src/
│   ├── detect_xenguocchieu.py    # Main detection & tracking
│   ├── detect_sailan.py          # Lane detection
│   └── train_yl11s.py            # Model training
├── model/
│   └── yolov11s.pt               # Pretrained YOLOv11s weights
├── configs/
│   └── data.yaml                 # Dataset configuration
└── README.md
`

## Requirements

`ash
pip install opencv-python numpy pandas matplotlib ultralytics
`

## Usage

### Detection

`python
python src/detect_xenguocchieu.py
`

## License

MIT License
