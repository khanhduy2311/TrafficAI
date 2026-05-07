import cv2
import os
import random
import glob

# ================= CẤU HÌNH CLASS ID =================
# Thay đổi các ID này cho khớp với file classes.txt của bạn
CAR_CLASS_ID = 0    # Lớp cần bị thay thế (Car)
BUS_CLASS_ID = 1    # Lớp thay thế 1 (Bus)
TRUCK_CLASS_ID = 2  # Lớp thay thế 2 (Truck)
# =======================================================

def yolo_to_pixel(yolo_bbox, img_width, img_height):
    """Chuyển đổi tọa độ YOLO (tương đối) sang tọa độ Pixel (tuyệt đối)"""
    class_id, cx, cy, w, h = yolo_bbox
    
    # Tính toán tọa độ x_min, y_min, x_max, y_max
    x_min = int((cx - w / 2) * img_width)
    y_min = int((cy - h / 2) * img_height)
    x_max = int((cx + w / 2) * img_width)
    y_max = int((cy + h / 2) * img_height)
    
    # Đảm bảo tọa độ không vượt quá kích thước ảnh
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(img_width, x_max), min(img_height, y_max)
    
    return x_min, y_min, x_max, y_max

def augment_copy_paste(img_path, label_path, replace_imgs_dir, output_img_path, output_label_path):
    """
    Hàm thực hiện việc tìm Car và thay bằng Bus/Truck
    """
    # 1. Đọc ảnh gốc
    img = cv2.imread(img_path)
    if img is None:
        print(f"Không thể đọc ảnh: {img_path}")
        return
    img_height, img_width = img.shape[:2]

    # 2. Chuẩn bị danh sách ảnh thay thế (Bus, Truck đã được crop sẵn)
    # Giả sử bạn có 1 thư mục chứa các ảnh crop của xe bus và xe tải
    replacement_images_paths = glob.glob(os.path.join(replace_imgs_dir, "*.jpg")) + \
                               glob.glob(os.path.join(replace_imgs_dir, "*.png"))
    
    if not replacement_images_paths:
        print("Cảnh báo: Không tìm thấy ảnh thay thế trong thư mục!")
        return

    # 3. Đọc file annotation (YOLO format)
    if not os.path.exists(label_path):
        return
    
    with open(label_path, 'r') as f:
        lines = f.readlines()

    new_annotations = []
    
    # 4. Duyệt qua từng object trong ảnh
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5: continue
        
        class_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])

        # Nếu phát hiện lớp Car (lớp cần thay thế)
        if class_id == CAR_CLASS_ID:
            # Xác suất 50% sẽ thay thế (bạn có thể chỉnh tỷ lệ này để không mất hết Car)
            if random.random() < 0.5: 
                # Chuyển đổi tọa độ
                x1, y1, x2, y2 = yolo_to_pixel((class_id, cx, cy, w, h), img_width, img_height)
                
                # Bỏ qua nếu bounding box quá nhỏ hoặc bị lỗi
                if x2 <= x1 or y2 <= y1:
                    new_annotations.append(line)
                    continue

                # --- BẮT ĐẦU QUÁ TRÌNH COPY - PASTE ---
                
                # a. Chọn ngẫu nhiên 1 ảnh Bus hoặc Truck
                repl_path = random.choice(replacement_images_paths)
                repl_img = cv2.imread(repl_path)
                
                # Quyết định class id mới dựa vào tên file hoặc thư mục (Ở đây code gán random cho ví dụ)
                # Tốt nhất bạn nên chia 2 thư mục bus_dir và truck_dir riêng rẽ
                new_class_id = random.choice([BUS_CLASS_ID, TRUCK_CLASS_ID]) 

                # b. Resize ảnh thay thế cho VỪA KHÍT với kích thước Bounding Box của Car cũ
                target_width = x2 - x1
                target_height = y2 - y1
                try:
                    repl_img_resized = cv2.resize(repl_img, (target_width, target_height))
                    
                    # c. Dán (Paste) đè lên ảnh gốc
                    img[y1:y2, x1:x2] = repl_img_resized
                    
                    # d. Lưu lại annotation mới
                    new_line = f"{new_class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                    new_annotations.append(new_line)
                    
                except Exception as e:
                    print(f"Lỗi khi resize/paste: {e}")
                    new_annotations.append(line) # Trữ lại nhãn cũ nếu lỗi
            else:
                # Giữ nguyên Car
                new_annotations.append(line)
        else:
            # Các lớp khác không phải Car thì giữ nguyên
            new_annotations.append(line)

    # 5. Lưu ảnh và file txt mới
    cv2.imwrite(output_img_path, img)
    with open(output_label_path, 'w') as f:
        f.writelines(new_annotations)

# ================= VÍ DỤ CÁCH SỬ DỤNG =================
if __name__ == "__main__":
    # Tạo các thư mục giả định (bạn tự thay đổi đường dẫn của bạn)
    os.makedirs("output_images", exist_ok=True)
    os.makedirs("output_labels", exist_ok=True)
    
    # Gọi hàm
    augment_copy_paste(
        img_path="data/images/001.jpg",              # Đường dẫn ảnh gốc chứa Car
        label_path="data/labels/001.txt",            # File label YOLO gốc
        replace_imgs_dir="data/source_bus_trucks",   # Thư mục chứa các ảnh xe Bus, Truck CẮT SẴN (chỉ có xe, không nền hoặc ít nền)
        output_img_path="output_images/001_aug.jpg", # Nơi lưu ảnh sau khi xử lý
        output_label_path="output_labels/001_aug.txt"# Nơi lưu label mới
    )
    print("Hoàn thành quá trình Copy-Paste Augmentation!")
