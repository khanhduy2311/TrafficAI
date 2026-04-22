"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm


def resize_image_and_update_yolo_labels(image_path, label_path, output_image_path, output_label_path, max_size=640):
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            if max(w, h) <= max_size:
                img.save(output_image_path)  # still copy to new dir
                # Copy label file as is
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        content = f.read()
                    with open(output_label_path, 'w') as f:
                        f.write(content)
                return True  # copied but not resized

            scale = max_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img.save(output_image_path)

            # Update YOLO labels - YOLO format uses normalized coordinates so no scaling needed
            # But we still copy the file
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    content = f.read()
                with open(output_label_path, 'w') as f:
                    f.write(content)

            return True

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


def resize_yolo_dataset(data_dir, max_size=640, num_workers=4):
    image_dir = os.path.join(data_dir, "images")
    label_dir = os.path.join(data_dir, "labels")
    
    output_image_dir = os.path.join(data_dir, "images_resized")
    output_label_dir = os.path.join(data_dir, "labels_resized")
    
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    if not os.path.exists(image_dir):
        print(f"Error: Images directory not found at {image_dir}")
        return
    
    if not os.path.exists(label_dir):
        print(f"Error: Labels directory not found at {label_dir}")
        return

    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]
    
    print(f"Found {len(image_files)} images to process")

    def process_image(image_file):
        # Get corresponding label file
        base_name = os.path.splitext(image_file)[0]
        label_file = base_name + '.txt'
        
        input_image_path = os.path.join(image_dir, image_file)
        input_label_path = os.path.join(label_dir, label_file)
        output_image_path = os.path.join(output_image_dir, image_file)
        output_label_path = os.path.join(output_label_dir, label_file)
        
        success = resize_image_and_update_yolo_labels(
            input_image_path, input_label_path, output_image_path, output_label_path, max_size
        )
        
        if success:
            return f"Processed: {image_file}"
        else:
            return f"Failed: {image_file}"

    # Use tqdm to track progress
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_image, image_file) for image_file in image_files]
        for future in tqdm(futures, total=len(futures), desc="Resizing images", unit="img"):
            results.append(future.result())

    # Print results
    successful = sum(1 for r in results if r.startswith("Processed:"))
    failed = sum(1 for r in results if r.startswith("Failed:"))
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} images")
    print(f"Failed: {failed} images")
    print(f"Output directories:")
    print(f"  Images: {output_image_dir}")
    print(f"  Labels: {output_label_dir}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Resize YOLO dataset images and update labels."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing images/ and labels/ folders",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=640,
        help="Maximum size for the longer side of the image (default: 640)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel threads (default: 4)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    resize_yolo_dataset(
        data_dir=args.data_dir,
        max_size=args.max_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()