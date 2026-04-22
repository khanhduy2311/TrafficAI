import cv2
import argparse
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Tool tra cuu toa do Polylines tren Camera")
    parser.add_argument("--source", required=True, help="Duong dan video hinh anh can do toa do")
    parser.add_argument("--output-config", default="configs/roi_config.json", help="Duong dan file JSON de luu cau hinh (mac dinh: configs/roi_config.json)")
    return parser.parse_args()

points = []
roi_list = [] # Luu danh sach cac ROI da ve

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        # Click chuot trai de them diem
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Click chuot phai de xoa diem gan nhat
        if len(points) > 0:
            points.pop()

def main():
    global points, roi_list
    args = parse_args()
    
    source_path = Path(args.source).resolve()
    if not source_path.exists():
        print(f"Error: Khong tim thay file {source_path}")
        return

    # Kiem tra file la anh hay video
    is_video = source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']

    if is_video:
        cap = cv2.VideoCapture(str(source_path))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Loi: Khong the do frame dau tien tu video.")
            return
    else:
        frame = cv2.imread(str(source_path))
        if frame is None:
             print("Loi: khong the doc hinh anh.")
             return

    # Tao ban sao de ve mousedown ma khong lam hu hinh goc
    display_frame = frame.copy()
    
    cv2.namedWindow('Draw ROI Polygon')
    cv2.setMouseCallback('Draw ROI Polygon', click_event)

    print("====================================")
    print(" HUONG DAN CHON VUNG ROI (3 Vung) ")
    print(" 1. Click CHUOT TRAI de cham diem")
    print(" 2. Click CHUOT PHAI de xoa diem vua cham")
    print(" 3. An phim 'C' de xoa toan bo")
    print(" 4. An phim 'N' de KET THUC ROI hien tai, CHUYEN SANG ROI TIEP THEO")
    print("    - ROI 1: Vung cho (Xanh bien)")
    print("    - ROI 2: Vung vi pham (Do)")
    print("    - ROI 3: Vung quet den (Trang)")
    print(" 5. An phim 'ENTER' hoac 'Q' de xac nhan & Luu tat ca ROI")
    print("====================================")

    while True:
        display_frame = frame.copy()
        
        # Ve cac ROI da luu (nhieu ROI truoc do)
        for idx, saved_roi in enumerate(roi_list):
            if idx == 0: color = (255, 0, 0)      # ROI 1: Blue (Wait)
            elif idx == 1: color = (0, 0, 255)    # ROI 2: Red (Violation)
            else: color = (255, 255, 255)         # ROI 3: White (Light)
            for _p in saved_roi:
                 cv2.circle(display_frame, _p, 5, color, -1)
            if len(saved_roi) > 1:
                for i in range(len(saved_roi) - 1):
                    cv2.line(display_frame, saved_roi[i], saved_roi[i+1], color, 2)
                cv2.line(display_frame, saved_roi[-1], saved_roi[0], color, 2)

        # Ve cac diem & duong noi hien tai
        for _p in points:
            cv2.circle(display_frame, _p, 5, (0, 255, 0), -1)
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(display_frame, points[i], points[i+1], (0, 255, 0), 2)
            cv2.line(display_frame, points[-1], points[0], (0, 255, 0), 2) # Dong da giac

        cv2.imshow('Draw ROI Polygon', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 13: # 13 la nut Enter
            if len(points) >= 3:
                 roi_list.append(list(points)) # Luu ROI cuoi cung neu chua an 'n'
            break
        elif key == ord('c'):
            points.clear()
        elif key == ord('n'):
            if len(points) >= 3:
                roi_list.append(list(points))
                points.clear()
                print(f"Da xac nhan ROI {len(roi_list)}. Tiep tuc ve ROI thu {len(roi_list) + 1}...")
            else:
                 print("Can it nhat 3 diem de tao 1 ROI!")

    cv2.destroyAllWindows()

    if len(roi_list) < 2:
        print("Ban chua ve du it nhat 2 ROI (Zone 1 va Zone 2). Kiem tra lai!")
    else:
        coords_roi1 = [str(c) for p in roi_list[0] for c in p]
        coords_roi2 = [str(c) for p in roi_list[1] for c in p]
        r1_str = ",".join(coords_roi1)
        r2_str = ",".join(coords_roi2)
        
        # Xu ly Light ROIs (Vung 3 tro di)
        light_rois_list = []
        light_rois_raw = []
        if len(roi_list) >= 3:
            for i in range(2, len(roi_list)):
                coords = [str(c) for p in roi_list[i] for c in p]
                light_rois_raw.append(",".join(coords))
                light_rois_list.append(f'"{",".join(coords)}"')
        
        # Luu vao JSON
        config_data = {
            "source": str(args.source),
            "roi1": r1_str,
            "roi2": r2_str,
            "light_roi": light_rois_raw,
            "updated_at": Path(args.source).stat().st_mtime if Path(args.source).exists() else 0
        }
        
        output_path = Path(args.output_config)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)
        
        print("\n" + "="*50)
        print(f" DA LUU CAU HINH VAO: {output_path}")
        print("="*50)

        light_cmd = ""
        if light_rois_list:
            light_cmd = " --light-roi " + " ".join(light_rois_list)
        
        print("\nHoac ban co the chay lenh thu cong:")
        print(f"python scripts/detect.py --source \"{args.source}\" --roi1 \"{r1_str}\" --roi2 \"{r2_str}\"{light_cmd}")

if __name__ == "__main__":
    main()