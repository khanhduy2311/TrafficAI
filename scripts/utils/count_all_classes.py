import os
from pathlib import Path
from collections import Counter

def count_all_classes(dataset_path):
    dataset_path = Path(dataset_path)
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        split_dir = dataset_path / split
        if not split_dir.exists() and split == 'valid':
            split_dir = dataset_path / 'val'
            
        if not split_dir.exists():
            continue
            
        label_dir = split_dir / 'labels'
        if not label_dir.exists():
            print(f"No labels in {split_dir}")
            continue
            
        counter = Counter()
        file_count = 0
        for label_file in label_dir.glob("*.txt"):
            file_count += 1
            with open(label_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    continue
                for line in content.split('\n'):
                    parts = line.split()
                    if parts:
                        counter[parts[0]] += 1
        
        print(f"\nSplit: {split} ({file_count} files)")
        for cls, count in sorted(counter.items()):
            print(f"  Class {cls}: {count}")

if __name__ == "__main__":
    datasets = [
        ("Data_new2", r"D:\Detect\Data_new2"),
        ("Datav2", r"D:\Detect\Datav2"),
        ("Datav2_Balanced", r"D:\Detect\Datav2_Balanced"),
        ("Datav2_Single", r"D:\Detect\Datav2_Single")
    ]
    for name, path in datasets:
        print(f"\n--- Detailed counts for {name} ---")
        count_all_classes(path)
