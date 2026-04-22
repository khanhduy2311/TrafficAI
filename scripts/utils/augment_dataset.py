import os
import cv2
import shutil
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm import tqdm

def get_train_augmentation(aug_type):
    """
    Tra ve cac phep bien doi khac nhau dua tren nhom anh.
    """
    # Them min_visibility=0.3 de loai bo cac box bi cat qua nhieu sau khi xoay
    bbox_params = A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3)
    
    if aug_type == "flip_brightness":
        return A.Compose([
            A.HorizontalFlip(p=1.0),
            A.RandomBrightnessContrast(p=0.5),
        ], bbox_params=bbox_params)
    
    elif aug_type == "rotate_noise":
        return A.Compose([
            # Dung Rotate thuan tuy (5-10 do) theo y ban
            A.Rotate(limit=10, p=1.0),
            A.GaussNoise(p=0.5),
        ], bbox_params=bbox_params)
        
    elif aug_type == "flip_rotate_blur":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=7, p=1.0),
            A.Blur(blur_limit=3, p=0.5),
        ], bbox_params=bbox_params)
    
    return None

def read_yolo_labels(path):
    bboxes = []
    class_labels = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    # YOLO: cls x y w h
                    bboxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                    class_labels.append(cls)
    return bboxes, class_labels

def save_yolo_labels(path, bboxes, class_labels):
    with open(path, 'w') as f:
        for cls, bbox in zip(class_labels, bboxes):
            f.write(f"{int(cls)} {' '.join([f'{x:.6f}' for x in bbox])}\n")

def augment_dataset(src_root, dst_root):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    
    # Tao cau truc thu muc moi
    for split in ['train', 'valid', 'test']:
        (dst_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (dst_root / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 1. COPY VALID & TEST (Giu nguyen ban)
    print("Copying valid and test splits (no augmentation)...")
    for split in ['valid', 'test']:
        for img_file in tqdm(list((src_root / split / 'images').glob('*'))):
            shutil.copy(img_file, dst_root / split / 'images' / img_file.name)
            lb_file = src_root / split / 'labels' / (img_file.stem + '.txt')
            if lb_file.exists():
                shutil.copy(lb_file, dst_root / split / 'labels' / lb_file.name)

    # 2. PHAN LOAI & AUGMENT TRAIN
    print("\nProcessing train split with balanced strategy...")
    train_img_dir = src_root / 'train' / 'images'
    train_lb_dir = src_root / 'train' / 'labels'
    
    yellow_imgs = []
    off_only_imgs = []
    normal_imgs = []

    all_train_imgs = list(train_img_dir.glob('*'))
    for img_file in all_train_imgs:
        lb_file = train_lb_dir / (img_file.stem + '.txt')
        if not lb_file.exists():
            normal_imgs.append(img_file)
            continue
            
        with open(lb_file, 'r') as f:
            classes = set(int(line.split()[0]) for line in f if line.strip())
            
        if 3 in classes:
            yellow_imgs.append(img_file)
        elif 1 in classes:
            off_only_imgs.append(img_file)
        else:
            normal_imgs.append(img_file)

    print(f"Groups: Yellow={len(yellow_imgs)}, Off-only={len(off_only_imgs)}, Normal={len(normal_imgs)}")

    # Thuc hien Augment
    # Nhóm Normal (x1)
    for img_file in tqdm(normal_imgs, desc="Processing Normal"):
        shutil.copy(img_file, dst_root / 'train' / 'images' / img_file.name)
        lb_file = train_lb_dir / (img_file.stem + '.txt')
        if lb_file.exists():
            shutil.copy(lb_file, dst_root / 'train' / 'labels' / lb_file.name)

    # Nhóm Off-only (x2)
    off_aug = get_train_augmentation("flip_brightness")
    for img_file in tqdm(off_only_imgs, desc="Processing Off (x2)"):
        # Copy ban goc
        dst_img_name = img_file.name
        shutil.copy(img_file, dst_root / 'train' / 'images' / dst_img_name)
        lb_file = train_lb_dir / (img_file.stem + '.txt')
        shutil.copy(lb_file, dst_root / 'train' / 'labels' / (img_file.stem + '.txt'))
        
        # Tao 1 ban augment
        image = cv2.imread(str(img_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes, labels = read_yolo_labels(lb_file)
        
        try:
            augmented = off_aug(image=image, bboxes=bboxes, class_labels=labels)
            
            # Kiem tra neu sau augment van con bboxes thi moi luu
            if augmented['bboxes']:
                aug_img_name = f"aug_{img_file.stem}_v1{img_file.suffix}"
                cv2.imwrite(str(dst_root / 'train' / 'images' / aug_img_name), cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR))
                save_yolo_labels(dst_root / 'train' / 'labels' / (f"aug_{img_file.stem}_v1.txt"), augmented['bboxes'], augmented['class_labels'])
        except Exception as e:
            print(f"Error augmenting {img_file.name}: {e}")

    # Nhóm Yellow (x4)
    yellow_augs = [
        get_train_augmentation("flip_brightness"),
        get_train_augmentation("rotate_noise"),
        get_train_augmentation("flip_rotate_blur")
    ]
    for img_file in tqdm(yellow_imgs, desc="Processing Yellow (x4)"):
        # Copy ban goc
        shutil.copy(img_file, dst_root / 'train' / 'images' / img_file.name)
        lb_file = train_lb_dir / (img_file.stem + '.txt')
        shutil.copy(lb_file, dst_root / 'train' / 'labels' / (img_file.stem + '.txt'))
        
        # Tao 3 ban augment
        image = cv2.imread(str(img_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes, labels = read_yolo_labels(lb_file)
        
        for i, aug in enumerate(yellow_augs):
            try:
                augmented = aug(image=image, bboxes=bboxes, class_labels=labels)
                
                # Kiem tra neu sau augment van con bboxes thi moi luu
                if augmented['bboxes']:
                    aug_id = i + 1
                    aug_img_name = f"aug_{img_file.stem}_v{aug_id}{img_file.suffix}"
                    cv2.imwrite(str(dst_root / 'train' / 'images' / aug_img_name), cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR))
                    save_yolo_labels(dst_root / 'train' / 'labels' / (f"aug_{img_file.stem}_v{aug_id}.txt"), augmented['bboxes'], augmented['class_labels'])
            except Exception as e:
                pass # Bo qua nhung tam bi loi toa do sau xoay (hiem)

    print("\nSUCCESS: Balanced dataset created at D:/Detect/Datav2_Balanced")

if __name__ == "__main__":
    augment_dataset("D:/Detect/Datav2", "D:/Detect/Datav2_Balanced")
