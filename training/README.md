# Training

Thư mục này chứa pipeline phục vụ huấn luyện mô hình nhận diện biển báo tốc độ trên dữ liệu fisheye.

## Cấu trúc thư mục

```text
training/
├─ configs/
│  └─ dataset.example.yaml
├─ scripts/
│  ├─ eval_models.py
│  └─ train_yolo11m.py
├─ weight/
│  └─ yolo11m_speedlimit.pt
├─ README.md
└─ split_dataset.py
```

## Cài đặt

Cài đặt các thư viện cần thiết từ thư mục gốc của project:

```bash
pip install -r requirements/training.txt
```

## Định dạng dữ liệu

Dữ liệu cần theo format YOLO, ví dụ:

```text
data_fisheye/
├─ images/
├─ labels/
├─ classes.txt
├─ train.txt
├─ val.txt
└─ test.txt
```

Bạn có thể bắt đầu từ file mẫu:

- `configs/dataset.example.yaml`

## Chia dữ liệu

Script `split_dataset.py` dùng để:

- tạo tập `train / val / test`
- sinh file `dataset.yaml`
- sinh file `dataset_balanced.yaml`
- tạo `train_balanced.txt` để hỗ trợ giảm mất cân bằng lớp
- xuất `split_stats.yaml` để thống kê phân bố dữ liệu

Cách chạy:

```bash
python training/split_dataset.py --data-dir /path/to/data_fisheye
```

Sau khi chạy, thư mục dataset sẽ có thêm:

- `train.txt`
- `val.txt`
- `test.txt`
- `train_balanced.txt`
- `dataset.yaml`
- `dataset_balanced.yaml`
- `split_stats.yaml`

Nếu tồn tại `dataset_balanced.yaml`, script train sẽ ưu tiên sử dụng file này.

## Huấn luyện mô hình

Hiện tại thư mục này đang dùng script huấn luyện chính:

```bash
python training/scripts/train_yolo11m.py
```

## Biến môi trường tùy chọn

Bạn có thể override cấu hình mặc định bằng các biến môi trường sau:

- `YOLO_DATA_CONFIG`: đường dẫn tới `dataset.yaml` hoặc `dataset_balanced.yaml`
- `YOLO_MODEL_PATH`: đường dẫn tới trọng số pretrained
- `YOLO_DEVICE`: GPU id, ví dụ `0`
- `YOLO_PROJECT_DIR`: thư mục lưu kết quả train

Ví dụ:

```bash
export YOLO_DATA_CONFIG=/path/to/data_fisheye/dataset_balanced.yaml
export YOLO_MODEL_PATH=training/weight/yolo11m_speedlimit.pt
export YOLO_DEVICE=0
export YOLO_PROJECT_DIR=./runs
python training/scripts/train_yolo11m.py
```

## Đánh giá mô hình

Sau khi train xong, bạn có thể đánh giá checkpoint bằng:

```bash
python training/scripts/eval_models.py --weights /path/to/runs --data /path/to/dataset.yaml
```
