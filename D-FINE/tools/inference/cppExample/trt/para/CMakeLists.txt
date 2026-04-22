cmake_minimum_required(VERSION 3.10)

project(
    DualDFINEInference
    LANGUAGES CXX CUDA
    VERSION 1.0.0
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV and CUDA
find_package(OpenCV REQUIRED core imgcodecs imgproc)
find_package(CUDA REQUIRED)
find_package(jsoncpp REQUIRED)

# Directly link to the TensorRT libraries without finding them
set(TRT_LIB_DIR "/usr/lib/x86_64-linux-gnu")

# Main executable
add_executable(
    dual_dfine_inference
    dual_dfine_inference.cpp
)

# Include directories - with explicit jsoncpp path
target_include_directories(
    dual_dfine_inference 
    PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIRS}
    "/usr/include"
    "/usr/include/x86_64-linux-gnu"
    "/usr/include/jsoncpp"  # Add this line for JsonCpp headers
)

# Direct library linking
target_link_libraries(
    dual_dfine_inference
    ${TRT_LIB_DIR}/libnvinfer.so.10
    ${TRT_LIB_DIR}/libnvonnxparser.so.10
    ${TRT_LIB_DIR}/libnvinfer_plugin.so.10
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    jsoncpp
)