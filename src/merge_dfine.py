#!/usr/bin/env python3

import os
import shutil
from pathlib import Path
import argparse
import concurrent.futures
import threading
from tqdm import tqdm

# Lock for thread-safe counter updates
counter_lock = threading.Lock()

def copy_file(src_path, dst_path, file_type, stats):
    """Copy a single file and update statistics."""
    try:
        if os.path.exists(dst_path):
            # Skip duplicate file - only keep the first occurrence
            with counter_lock:
                stats[file_type]["duplicates"] += 1
            return True  # Return True to indicate we handled it (by skipping)
        else:
            shutil.copy2(src_path, dst_path)
        
        with counter_lock:
            stats[file_type]["total"] += 1
            
        return True
    except Exception as e:
        print(f"Lỗi khi copy file {src_path}: {e}")
        return False

def merge_datasets(batch1_path, batch2_path, output_path, num_workers=4):
    """
    Gộp hai thư mục dataset với cấu trúc YOLO (images và labels) thành một thư mục mới.
    Khi có file trùng lặp, chỉ giữ lại file đầu tiên.
    
    Args:
        batch1_path: Đường dẫn đến thư mục batch_1
        batch2_path: Đường dẫn đến thư mục batch_2
        output_path: Đường dẫn đến thư mục đích để lưu dataset đã gộp
        num_workers: Số luồng xử lý song song
    """
    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels"), exist_ok=True)
    
    # Danh sách các batch cần xử lý
    batches = [
        {"name": "batch_1", "path": batch1_path},
        {"name": "batch_2", "path": batch2_path}
    ]
    
    # Thống kê số lượng file đã xử lý
    stats = {
        "images": {"total": 0, "duplicates": 0},
        "labels": {"total": 0, "duplicates": 0}
    }
    
    # Tạo danh sách các tác vụ cần thực hiện
    copy_tasks = []
    
    for batch in batches:
        batch_path = batch["path"]
        
        # Đường dẫn đến thư mục images và labels trong batch hiện tại
        images_path = os.path.join(batch_path, "train", "images")
        labels_path = os.path.join(batch_path, "train", "labels")
        
        # Kiểm tra thư mục tồn tại
        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            print(f"Warning: Không tìm thấy thư mục train/images hoặc train/labels trong {batch['name']}")
            continue
        
        print(f"Chuẩn bị xử lý files từ {batch['name']}...")
        
        # Chuẩn bị tác vụ copy ảnh
        for image_file in os.listdir(images_path):
            src_path = os.path.join(images_path, image_file)
            dst_path = os.path.join(output_path, "images", image_file)
            copy_tasks.append((src_path, dst_path, "images", stats))
        
        # Chuẩn bị tác vụ copy nhãn
        for label_file in os.listdir(labels_path):
            src_path = os.path.join(labels_path, label_file)
            dst_path = os.path.join(output_path, "labels", label_file)
            copy_tasks.append((src_path, dst_path, "labels", stats))
    
    # Thực hiện song song các tác vụ copy
    total_files = len(copy_tasks)
    print(f"Bắt đầu xử lý {total_files} files với {num_workers} luồng...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Sử dụng tqdm để hiển thị tiến trình
        results = list(tqdm(
            executor.map(lambda args: copy_file(*args), copy_tasks),
            total=total_files,
            desc="Đang copy files"
        ))
    
    # In thống kê
    print("\nHoàn thành gộp datasets:")
    print(f"Tổng số ảnh đã copy: {stats['images']['total']} (bỏ qua {stats['images']['duplicates']} trùng lặp)")
    print(f"Tổng số nhãn đã copy: {stats['labels']['total']} (bỏ qua {stats['labels']['duplicates']} trùng lặp)")
    print(f"Dataset mới đã được lưu tại: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gộp hai thư mục dataset YOLO thành một, chỉ giữ lại file đầu tiên khi trùng lặp")
    parser.add_argument("--batch1", type=str, default="./batch_1", help="Đường dẫn đến thư mục batch_1")
    parser.add_argument("--batch2", type=str, default="./batch_2", help="Đường dẫn đến thư mục batch_2")
    parser.add_argument("--output", type=str, default="./merged_dataset", help="Đường dẫn đến thư mục đầu ra")
    parser.add_argument("--workers", type=int, default=25, help="Số luồng xử lý song song")
    
    args = parser.parse_args()
    
    merge_datasets(args.batch1, args.batch2, args.output, args.workers)