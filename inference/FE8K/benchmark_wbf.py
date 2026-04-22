# benchmark_wbf.py
#
# A script to rigorously test and benchmark our C++ WBF implementation against
# the original `ensemble-boxes` library using a large, realistic dataset.

import numpy as np
import torch  # Must be imported first to load C++ dependencies
import time

# Import the library we are comparing against
from ensemble_boxes import weighted_boxes_fusion

# Import our newly compiled C++ module
try:
    from fusion import wbf_cpp
except ImportError as e:
    print(f"\n--- FATAL ERROR ---")
    print(f"Could not import the C++ WBF module. Error: {e}")
    print("Please ensure you have run: cd fusion; python setup_wbf.py build_ext --inplace; cd ..")
    exit(1)


def generate_test_data(num_total_boxes, num_models, num_classes):
    """
    Generates a realistic, large-scale test dataset for WBF.
    """
    print(f"Generating test data with ~{num_total_boxes} boxes across {num_models} models...")
    
    boxes_list = []
    scores_list = []
    labels_list = []
    
    boxes_per_model = num_total_boxes // num_models

    for _ in range(num_models):
        # Generate random coordinates for two points
        p1 = np.random.rand(boxes_per_model, 2)
        p2 = np.random.rand(boxes_per_model, 2)
        
        # Ensure x1 < x2 and y1 < y2 to create valid boxes
        boxes = np.hstack([
            np.minimum(p1[:, 0:1], p2[:, 0:1]),
            np.minimum(p1[:, 1:2], p2[:, 1:2]),
            np.maximum(p1[:, 0:1], p2[:, 0:1]),
            np.maximum(p1[:, 1:2], p2[:, 1:2]),
        ]).astype(np.float32)
        
        scores = np.random.rand(boxes_per_model).astype(np.float32)
        labels = np.random.randint(0, num_classes, size=boxes_per_model).astype(np.int32)
        
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)
        
    return boxes_list, scores_list, labels_list


def run_benchmark(wbf_function, boxes, scores, labels, iou_thr, skip_box_thr, weights, num_iterations):
    """
    Times a given WBF function over many iterations and returns the average time per call.
    """
    # Run once for warmup (e.g., for Numba JIT compilation)
    wbf_function(boxes, scores, labels, iou_thr=iou_thr, skip_box_thr=skip_box_thr, weights=weights)

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        _ = wbf_function(boxes, scores, labels, iou_thr=iou_thr, skip_box_thr=skip_box_thr, weights=weights)
    end_time = time.perf_counter()
    
    avg_time_ms = (end_time - start_time) * 1000 / num_iterations
    return avg_time_ms


def main():
    # --- Configuration ---
    NUM_TOTAL_BOXES = 300
    NUM_MODELS = 2
    NUM_CLASSES = 5
    IOU_THR = 0.55
    SKIP_BOX_THR = 0.1
    WEIGHTS = [1.0] * NUM_MODELS
    BENCHMARK_ITERATIONS = 100

    # 1. Generate a large, random dataset
    boxes_list, scores_list, labels_list = generate_test_data(NUM_TOTAL_BOXES, NUM_MODELS, NUM_CLASSES)
    
    print("\n--- Correctness Validation (Large Dataset) ---")
    
    # 2. Run both implementations once to check for correctness
    py_boxes, py_scores, py_labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=WEIGHTS, iou_thr=IOU_THR, skip_box_thr=SKIP_BOX_THR)
    cpp_boxes, cpp_scores, cpp_labels = wbf_cpp.fusion(boxes_list, scores_list, labels_list, IOU_THR, SKIP_BOX_THR, WEIGHTS)

    try:
        # Sort for consistent comparison
        py_sort_indices = np.argsort(py_scores)[::-1]
        cpp_sort_indices = np.argsort(cpp_scores)[::-1]
        
        assert len(py_boxes) == len(cpp_boxes), "Mismatch in number of fused boxes!"
        assert np.allclose(py_boxes[py_sort_indices], cpp_boxes[cpp_sort_indices], atol=1e-4), "Fused BOXES do not match!"
        assert np.allclose(py_scores[py_sort_indices], cpp_scores[cpp_sort_indices], atol=1e-4), "Fused SCORES do not match!"
        assert np.array_equal(py_labels[py_sort_indices].astype(int), cpp_labels[cpp_sort_indices].astype(int)), "Fused LABELS do not match!"
        
        print("✅ SUCCESS: C++ implementation is correct on the large dataset.")
    except AssertionError as e:
        print(f"❌ FAILURE: Correctness check failed on large dataset. {e}")
        return # Stop if correctness fails

    # 3. If correctness passes, run the performance benchmark
    print(f"\n--- Performance Benchmark ({BENCHMARK_ITERATIONS} iterations) ---")

    python_time = run_benchmark(weighted_boxes_fusion, boxes_list, scores_list, labels_list, IOU_THR, SKIP_BOX_THR, WEIGHTS, BENCHMARK_ITERATIONS)
    cpp_time = run_benchmark(wbf_cpp.fusion, boxes_list, scores_list, labels_list, IOU_THR, SKIP_BOX_THR, WEIGHTS, BENCHMARK_ITERATIONS)

    # 4. Report the results
    print("\n" + "="*40)
    print("           BENCHMARK RESULTS")
    print("="*40)
    print(f"Original (Python + Numba): {python_time:8.4f} ms per call")
    print(f"New      (C++ Extension) : {cpp_time:8.4f} ms per call")
    print("="*40)
    
    if cpp_time > 0:
        speedup = python_time / cpp_time
        print(f"\n✅ CONCLUSION: The C++ version is {speedup:.2f}x faster.")
    else:
        print("\nCONCLUSION: C++ version was too fast to measure accurately.")


if __name__ == "__main__":
    main()
