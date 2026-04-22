import os
from pathlib import Path
from collections import Counter

def identify_groups(dataset_path):
    p = Path(dataset_path) / 'train' / 'labels'
    yellow_group = []
    off_only_group = []
    normal_group = [] # Red/Green only
    
    for lb in p.glob('*.txt'):
        with open(lb, 'r') as f:
            classes = set(int(line.split()[0]) for line in f if line.strip())
        
        if 3 in classes: # Has Yellow
            yellow_group.append(lb.stem)
        elif 1 in classes: # Has Off but no Yellow
            off_only_group.append(lb.stem)
        else: # Red/Green only or Background
            normal_group.append(lb.stem)
            
    print(f"Yellow Group (x4 target): {len(yellow_group)} images")
    print(f"Off-only Group (x2 target): {len(off_only_group)} images")
    print(f"Normal Group (x1 target): {len(normal_group)} images")
    
    return yellow_group, off_only_group, normal_group

if __name__ == "__main__":
    identify_groups("D:/Detect/Datav2")
