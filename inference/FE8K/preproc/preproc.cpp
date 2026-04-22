// preproc.cpp
#include <torch/extension.h>
#include <pybind11/numpy.h> // <-- THIS IS THE FIX. Include the NumPy header.
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Helper to convert OpenCV Mat to a Torch Tensor
torch::Tensor mat_to_tensor(const cv::Mat& mat, const torch::Device& device) {
    // from_blob does not take ownership of the memory, so it's a zero-copy view from CPU Mat
    return torch::from_blob(mat.data, {mat.rows, mat.cols, mat.channels()}, torch::kUInt8)
        .to(device, torch::kFloat, /*non_blocking=*/true) // Move to GPU and convert to float asynchronously
        .permute({2, 0, 1})       // HWC -> CHW
        .div_(255.0);             // Normalize in-place
}

py::dict dfine_preprocess_cpp(py::array_t<uint8_t> img_bgr_py, const std::string& device_str) {
    // The 'py::gil_scoped_release' block is the magic. It tells the Python
    // interpreter that this block of code does not touch any Python objects,
    // so the GIL can be released for other threads to use.
    py::gil_scoped_release release;

    // Zero-copy conversion from NumPy array to OpenCV Mat
    py::buffer_info buf = img_bgr_py.request();
    cv::Mat img_bgr(buf.shape[0], buf.shape[1], CV_8UC3, (uint8_t*)buf.ptr);

    // Get original size
    int h = img_bgr.rows;
    int w = img_bgr.cols;

    // --- All CPU-bound work happens here, with the GIL released ---
    cv::Mat img_rgb, img_resized;
    cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);
    cv::resize(img_rgb, img_resized, cv::Size(960, 960), 0, 0, cv::INTER_AREA);

    // Re-acquire the GIL before creating PyTorch/Python objects to return
    py::gil_scoped_acquire acquire;

    torch::Device device(device_str);

    // Create tensors on the target device
    auto orig_size_tensor = torch::tensor({{w, h}}, torch::TensorOptions().device(device)); // Corrected to be 2D
    auto image_tensor = mat_to_tensor(img_resized, device).unsqueeze(0);

    // Create a Python dictionary to return the results
    py::dict result;
    result["images"] = image_tensor;
    result["orig_target_sizes"] = orig_size_tensor;

    return result;
}

// Macro to bind the C++ function to a Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("preprocess", &dfine_preprocess_cpp, "D-FINE pre-processing in C++");
}
