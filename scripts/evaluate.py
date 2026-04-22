import argparse
from pathlib import Path
from ultralytics import YOLO

def evaluate(weights, data, split='test', name=None):
    model = YOLO(weights)
    results = model.val(data=data, split=split, name=name)
    
    # Extract metrics
    precision = results.box.mp
    recall = results.box.mr
    map50 = results.box.map50
    map95 = results.box.map
    f1 = 2 * (precision * recall) / (precision + recall + 1e-16)
    
    return {
        "model": Path(weights).name,
        "data": Path(data).name,
        "split": split,
        "precision": f"{precision:.5f}",
        "recall": f"{recall:.5f}",
        "f1": f"{f1:.5f}",
        "map50": f"{map50:.5f}",
        "map95": f"{map95:.5f}",
        "weights": str(Path(weights).resolve()),
        "eval_dir": str(Path(results.save_dir).resolve())
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--name", default="eval_test")
    args = parser.parse_args()
    
    metrics = evaluate(args.weights, args.data, args.split, args.name)
    
    csv_line = f"{metrics['model']},{metrics['data']},{metrics['split']},{metrics['precision']},{metrics['recall']},{metrics['f1']},{metrics['map50']},{metrics['map95']},{metrics['weights']},{metrics['eval_dir']}"
    print("\n--- CSV ROW ---")
    print(csv_line)
    print("---------------")
    
    # Append to CSV
    csv_file = Path("outputs/results/model_eval_metrics.csv")
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    if not csv_file.exists():
        with open(csv_file, "w") as f:
            f.write("model,data,split,precision,recall,f1_score,mAP50,mAP50_95,weights,eval_dir\n")
            
    with open(csv_file, "a") as f:
        f.write(csv_line + "\n")
    
    print(f"Results appended to {csv_file}")

if __name__ == "__main__":
    main()
