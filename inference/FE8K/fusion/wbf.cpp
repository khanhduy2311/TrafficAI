// wbf.cpp (Final Definitive Version - No Internal Sorting)
#include <torch/extension.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <map>

// Using double for all floating-point operations to match Python/NumPy
struct InternalBox { int label; double score_with_weight; double weight; int model_index; double x1, y1, x2, y2; };

int find_matching_box_fast(const std::vector<InternalBox>& weighted_boxes, const InternalBox& new_box, double iou_thr) {
    if (weighted_boxes.empty()) return -1;
    int best_idx = -1; double best_iou = -1.0;
    for (size_t i = 0; i < weighted_boxes.size(); ++i) {
        double x_left = (std::max)(weighted_boxes[i].x1, new_box.x1); double y_top = (std::max)(weighted_boxes[i].y1, new_box.y1);
        double x_right = (std::min)(weighted_boxes[i].x2, new_box.x2); double y_bottom = (std::min)(weighted_boxes[i].y2, new_box.y2);
        if (x_right < x_left || y_bottom < y_top) continue;
        double intersection_area = (x_right - x_left) * (y_bottom - y_top);
        double box1_area = (weighted_boxes[i].x2 - weighted_boxes[i].x1) * (weighted_boxes[i].y2 - weighted_boxes[i].y1);
        double box2_area = (new_box.x2 - new_box.x1) * (new_box.y2 - new_box.y1);
        double union_area = box1_area + box2_area - intersection_area;
        if (union_area < 1e-9) continue;
        double iou = intersection_area / union_area;
        if (iou > best_iou) { best_iou = iou; best_idx = i; }
    }
    if (best_iou > iou_thr) return best_idx;
    return -1;
}

InternalBox get_weighted_box(const std::vector<InternalBox>& boxes) {
    double total_conf = 0, total_weight = 0, f_x1 = 0, f_y1 = 0, f_x2 = 0, f_y2 = 0;
    for (const auto& b : boxes) {
        total_conf += b.score_with_weight; total_weight += b.weight;
        f_x1 += b.score_with_weight * b.x1; f_y1 += b.score_with_weight * b.y1;
        f_x2 += b.score_with_weight * b.x2; f_y2 += b.score_with_weight * b.y2;
    }
    return {boxes[0].label, total_conf / boxes.size(), total_weight, -1, f_x1 / total_conf, f_y1 / total_conf, f_x2 / total_conf, f_y2 / total_conf};
}

py::tuple single_class_wbf_cpp(
    py::list box_list_py, // Now expects a single, pre-sorted list of boxes
    double iou_thr, py::object weights_py) {

    std::vector<InternalBox> box_list;
    std::vector<double> weights;
    if (weights_py.is_none()) { /* Should be handled in Python */ }
    else { weights = weights_py.cast<std::vector<double>>(); }

    // STEP A: Convert the pre-sorted Python list to C++ structs
    for (const auto& item : box_list_py) {
        py::tuple t = item.cast<py::tuple>();
        box_list.push_back({
            t[0].cast<int>(), t[1].cast<double>(), t[2].cast<double>(), t[3].cast<int>(),
            t[4].cast<double>(), t[5].cast<double>(), t[6].cast<double>(), t[7].cast<double>()
        });
    }

    // STEP B: Main fusion logic (no sorting!)
    std::vector<InternalBox> overall_boxes;
    {
        py::gil_scoped_release release;
        if (!box_list.empty()) {
            std::vector<std::vector<InternalBox>> new_clusters;
            std::vector<InternalBox> weighted_boxes;
            for (const auto& box : box_list) {
                int index = find_matching_box_fast(weighted_boxes, box, iou_thr);
                if (index != -1) {
                    new_clusters[index].push_back(box);
                    weighted_boxes[index] = get_weighted_box(new_clusters[index]);
                } else {
                    new_clusters.push_back({box});
                    weighted_boxes.push_back(box);
                }
            }
            double total_models_weight = 0.0;
            for(double w : weights) total_models_weight += w;
            size_t num_models = weights.size();
            for (size_t i = 0; i < new_clusters.size(); ++i) {
                double cluster_size = static_cast<double>(new_clusters[i].size());
                weighted_boxes[i].score_with_weight = weighted_boxes[i].score_with_weight * (std::min)((double)num_models, cluster_size) / total_models_weight;
                overall_boxes.push_back(weighted_boxes[i]);
            }
        }
    }

    // STEP C: Convert results back to Python
    std::sort(overall_boxes.begin(), overall_boxes.end(), [](const InternalBox& a, const InternalBox& b){ return a.score_with_weight > b.score_with_weight; });
    size_t num_fused = overall_boxes.size();
    py::array_t<double> res_boxes(std::vector<py::ssize_t>{num_fused, 4});
    py::array_t<double> res_scores(num_fused);
    py::array_t<int> res_labels(num_fused);
    auto res_boxes_ptr = res_boxes.mutable_unchecked<2>();
    auto res_scores_ptr = res_scores.mutable_unchecked<1>();
    auto res_labels_ptr = res_labels.mutable_unchecked<1>();
    for (size_t i = 0; i < num_fused; ++i) {
        res_boxes_ptr(i, 0) = overall_boxes[i].x1; res_boxes_ptr(i, 1) = overall_boxes[i].y1;
        res_boxes_ptr(i, 2) = overall_boxes[i].x2; res_boxes_ptr(i, 3) = overall_boxes[i].y2;
        res_scores_ptr(i) = overall_boxes[i].score_with_weight;
        res_labels_ptr(i) = overall_boxes[i].label;
    }
    return py::make_tuple(res_boxes, res_scores, res_labels);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fusion", &single_class_wbf_cpp, "Single-class WBF in C++ (Double Precision, Pre-sorted Input)",
        py::arg("box_list"),
        py::arg("iou_thr"),
        py::arg("weights") = py::none());
}
