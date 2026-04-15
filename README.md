# Hệ Thống Phát Hiện Vi Phạm Giao Thông (Tối Ưu Cho Camera Fisheye) - YOLO11

Một quy trình phát hiện đối tượng hiệu năng cao chuyên dùng để tự động xử lý các vi phạm giao thông (Vượt đèn đỏ) sử dụng **YOLO11**. Hệ thống này được tinh chỉnh đặc biệt cho các loại camera góc rộng (Fisheye) phổ biến trong giám sát giao thông đô thị.

## 🚀 Các Tính Năng Chính
- **Cơ Chế Dự Phòng Đèn Tín Hiệu (Redundancy)**: Hỗ trợ nhiều vùng quét đèn (Light ROIs) cùng lúc (Vd: Đèn bên trái và bên phải) để đảm bảo vẫn nhận diện được trạng thái đèn ngay cả khi một bên bị xe lớn che khuất.
- **Logic Xử Lý Cực Kỳ Chặt Chẽ**: Sử dụng cơ chế Zone-based (Vùng 1: Chờ vạch, Vùng 2: Vi phạm) kết hợp với thuật toán làm mượt trạng thái đèn và thời gian chờ (cooldown) để tránh bắt nhầm.
- **Huấn Luyện Hợp Nhất (Unified Training)**: Một engine huấn luyện tập trung với các bộ thiết lập (Presets) sẵn có cho dòng model Nano/Small và các cấu hình tối ưu dữ liệu.
- **Bằng Chứng Vi Phạm Toàn Diện**: Tự động lưu trữ ảnh crop xe, ảnh toàn cảnh (có mã hóa SHA256 để đảm bảo tính pháp lý) và video clip hành vi vi phạm.
- **Công Cụ Trực Quan**: Tích hợp công cụ GUI để vẽ các đa giác (Polygon) tùy chỉnh cho bất kỳ ngã tư nào chỉ với vài cú click.

---

## 🏗️ 3 Trụ Cột Chính

Để vận hành hệ thống, bạn sẽ chủ yếu tương tác với 3 file script sau:

### 1. Unified Trainer (`scripts/research/train_yolo.py`)
"Trái tim" huấn luyện của toàn bộ dự án. Nó chứa các siêu tham số (hyperparameters) và các cài đặt tăng cường dữ liệu đã được tối ưu.
```powershell
python scripts/research/train_yolo.py --preset optimized_legacy --data configs/data.yaml
```

### 2. ROI Setup Tool (`scripts/utils/get_polygon_roi.py`)
Sử dụng công cụ GUI này để vẽ làn đường và vùng quét đèn cho một góc camera cụ thể.
```powershell
python scripts/utils/get_polygon_roi.py --source detect.mp4
```
*   **Zone 1**: Vùng chờ trước vạch.
*   **Zone 2**: Vùng vi phạm.
*   **Zone 3+**: Các vùng quét đèn (Hỗ trợ đa vùng để dự phòng).

### 3. Detection Engine (`scripts/detect.py`)
File thực thi chính để phát hiện vi phạm từ video hoặc luồng camera trực tiếp.
```powershell
python scripts/detect.py --source "detect.mp4" --light-weights "models/best_collection/yolo11n_v2.pt" --roi1 "..." --roi2 "..." --light-roi "roi_1" "roi_2"
```

---

## 📊 Phân Tích & Đánh Giá
- **Bảng Xếp Hạng (Leaderboards)**: Sử dụng `scripts/utils/generate_leaderboard.py` để so sánh Precision/Recall/mAP giữa tất cả các model đã train.
- **Kiểm Tra Hiệu Năng (Inference Benchmark)**: Sử dụng `scripts/utils/infer_video.py` để đo tốc độ xử lý thực tế (FPS) trên phần cứng của bạn.

## 🛠️ Cài Đặt
1.  Clone repository này về máy.
2.  Cài đặt các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
    ```
3.  Đặt dữ liệu của bạn vào thư mục `Data/` (theo cấu trúc chuẩn của YOLO).

---

## 📂 Cấu Trúc Dự Án
- `scripts/`: Toàn bộ mã nguồn Python.
- `configs/`: Các file YAML cấu hình dataset và mô hình.
- `models/best_collection/`: Nơi lưu trữ các model tốt nhất sẵn sàng để triển khai.
- `outputs/`: Nơi tự động lưu log vi phạm và kết quả xử lý.

---
*Dành cho các hệ thống giám sát và xử lý vi phạm giao thông tự động.*
