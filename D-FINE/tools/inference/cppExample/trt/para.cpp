#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <filesystem>
#include <map>
#include <numeric>
#include <iomanip>
#include <json/json.h> // For JSON output

namespace fs = std::filesystem;

// Logger for TensorRT messages
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
    static TRTLogger& getInstance() {
        static TRTLogger logger;
        return logger;
    }
};

// Detection structure to store bounding box, class, and confidence
struct Detection {
    float x1, y1, x2, y2;  // Bounding box coordinates
    float score;           // Confidence score
    int class_id;          // Class ID

    Detection() : x1(0), y1(0), x2(0), y2(0), score(0), class_id(0) {}

    Detection(float x1_, float y1_, float x2_, float y2_, float score_, int class_id_)
        : x1(x1_), y1(y1_), x2(y2_), y2(y2_), score(score_), class_id(class_id_) {}

    float width() const { return x2 - x1; }
    float height() const { return y2 - y1; }
    float area() const { return width() * height(); }
};

// Performance timing utilities
class Timer {
public:
    void start() {
        startTime = std::chrono::high_resolution_clock::now();
    }

    double stop() {
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = endTime - startTime;
        return elapsed.count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
};

// TensorRT engine wrapper class
class TRTEngine {
public:
    TRTEngine(const std::string& engine_path, cudaStream_t stream)
        : stream(stream), runtime(nullptr), engine(nullptr), context(nullptr), buffers(nullptr) {

        // Load engine from file
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) {
            throw std::runtime_error("Failed to open engine file: " + engine_path);
        }

        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);

        std::vector<char> engineData(size);
        file.read(engineData.data(), size);
        file.close();

        // Create runtime and engine
        runtime = nvinfer1::createInferRuntime(TRTLogger::getInstance());
        engine = runtime->deserializeCudaEngine(engineData.data(), size);
        if (!engine) {
            throw std::runtime_error("Failed to create CUDA engine");
        }

        // Create execution context
        context = engine->createExecutionContext();
        if (!context) {
            throw std::runtime_error("Failed to create execution context");
        }

        // Allocate buffers
        allocateBuffers();
    }

    ~TRTEngine() {
        // Free allocated memory
        for (int i = 0; i < numBindings; i++) {
            if (buffers[i]) {
                cudaFree(buffers[i]);
            }
        }
        delete[] buffers;

        // Use proper deletion for TensorRT objects (instead of destroy())
        if (context) {
            delete context;  // Changed from context->destroy()
        }
        if (engine) {
            delete engine;   // Changed from engine->destroy()
        }
        if (runtime) {
            delete runtime;  // Changed from runtime->destroy()
        }
    }

    void infer(const float* input_data, const int64_t* image_size_data) {
        // Copy input data to device
        cudaMemcpyAsync(buffers[inputIndex], input_data, inputSize, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(buffers[imageSizeIndex], image_size_data, imageSizeBytes, cudaMemcpyHostToDevice, stream);

        // Execute inference - using enqueueV3 instead of enqueueV2
        context->enqueueV3(stream);  // Changed from enqueueV2(buffers, stream, nullptr)
    }

    std::vector<Detection> getDetections(float score_threshold = 0.0001, int max_detections = 300) {
        std::vector<Detection> detections;

        // Allocate host memory for results
        float* h_boxes = new float[boxesSize / sizeof(float)];
        float* h_scores = new float[scoresSize / sizeof(float)];
        int64_t* h_labels = new int64_t[labelsSize / sizeof(int64_t)];

        // Copy results from device to host
        cudaMemcpyAsync(h_boxes, buffers[boxesIndex], boxesSize, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_scores, buffers[scoresIndex], scoresSize, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_labels, buffers[labelsIndex], labelsSize, cudaMemcpyDeviceToHost, stream);

        // Wait for the stream to complete
        cudaStreamSynchronize(stream);

        // Process detections
        int num_boxes = boxesSize / sizeof(float) / 4; // Each box has 4 coordinates
        for (int i = 0; i < num_boxes && i < max_detections; i++) {
            if (h_scores[i] > score_threshold) {
                float x1 = h_boxes[i * 4];
                float y1 = h_boxes[i * 4 + 1];
                float x2 = h_boxes[i * 4 + 2];
                float y2 = h_boxes[i * 4 + 3];
                int class_id = static_cast<int>(h_labels[i]);

                detections.emplace_back(x1, y1, x2, y2, h_scores[i], class_id);
            }
        }

        // Free host memory
        delete[] h_boxes;
        delete[] h_scores;
        delete[] h_labels;

        return detections;
    }

private:
    void allocateBuffers() {
        numBindings = engine->getNbIOTensors();
        buffers = new void*[numBindings];

        // Find input and output bindings
        for (int i = 0; i < numBindings; i++) {
            const char* name = engine->getIOTensorName(i);
            nvinfer1::Dims dims = engine->getTensorShape(name);
            size_t size = 1;

            for (int j = 0; j < dims.nbDims; j++) {
                size *= dims.d[j];
            }

            // Determine the data type and size
            nvinfer1::DataType dtype = engine->getTensorDataType(name);
            size_t elementSize = 0;

            switch (dtype) {
                case nvinfer1::DataType::kFLOAT:
                    elementSize = sizeof(float);
                    break;
                case nvinfer1::DataType::kINT64:
                    elementSize = sizeof(int64_t);
                    break;
                case nvinfer1::DataType::kINT32:
                    elementSize = sizeof(int32_t);
                    break;
                default:
                    throw std::runtime_error("Unsupported data type");
            }

            size_t totalSize = size * elementSize;

            // Allocate GPU memory
            cudaMalloc(&buffers[i], totalSize);

            // For TensorRT 10+, set input/output bindings
            if (strcmp(name, "images") == 0) {
                inputIndex = i;
                inputSize = totalSize;
                context->setTensorAddress(name, buffers[i]);
            } else if (strcmp(name, "orig_target_sizes") == 0) {
                imageSizeIndex = i;
                imageSizeBytes = totalSize;
                context->setTensorAddress(name, buffers[i]);
            } else if (strcmp(name, "boxes") == 0) {
                boxesIndex = i;
                boxesSize = totalSize;
                context->setTensorAddress(name, buffers[i]);
            } else if (strcmp(name, "scores") == 0) {
                scoresIndex = i;
                scoresSize = totalSize;
                context->setTensorAddress(name, buffers[i]);
            } else if (strcmp(name, "labels") == 0) {
                labelsIndex = i;
                labelsSize = totalSize;
                context->setTensorAddress(name, buffers[i]);
            }
        }
    }

    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    cudaStream_t stream;

    void** buffers;
    int numBindings;
    int inputIndex, imageSizeIndex, boxesIndex, scoresIndex, labelsIndex;
    size_t inputSize, imageSizeBytes, boxesSize, scoresSize, labelsSize;
};

// Calculate IoU between two bounding boxes
float calculateIoU(const Detection& box1, const Detection& box2) {
    float x1 = std::max(box1.x1, box2.x1);
    float y1 = std::max(box1.y1, box2.y1);
    float x2 = std::min(box1.x2, box2.x2);
    float y2 = std::min(box1.y2, box2.y2);

    float intersectionArea = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float unionArea = box1.area() + box2.area() - intersectionArea;

    return unionArea > 0 ? intersectionArea / unionArea : 0;
}

// Weighted Box Fusion implementation
std::vector<Detection> applyWBF(
    const std::vector<Detection>& detections1,
    const std::vector<Detection>& detections2,
    float iou_threshold = 0.55,
    float score_threshold = 0.0001
) {
    // Combine detections from both models
    std::vector<Detection> all_detections;
    all_detections.insert(all_detections.end(), detections1.begin(), detections1.end());
    all_detections.insert(all_detections.end(), detections2.begin(), detections2.end());

    // Filter by score threshold
    std::vector<Detection> filtered_detections;
    for (const auto& det : all_detections) {
        if (det.score >= score_threshold) {
            filtered_detections.push_back(det);
        }
    }

    if (filtered_detections.empty()) {
        return {};
    }

    // Sort by score (descending)
    std::sort(filtered_detections.begin(), filtered_detections.end(),
        [](const Detection& a, const Detection& b) { return a.score > b.score; });

    std::vector<std::vector<int>> clusters;
    std::vector<bool> is_clustered(filtered_detections.size(), false);

    // Cluster boxes with high IoU
    for (size_t i = 0; i < filtered_detections.size(); i++) {
        if (is_clustered[i]) continue;

        std::vector<int> cluster;
        cluster.push_back(i);
        is_clustered[i] = true;

        for (size_t j = i + 1; j < filtered_detections.size(); j++) {
            if (is_clustered[j]) continue;

            // Only cluster boxes of the same class
            if (filtered_detections[i].class_id != filtered_detections[j].class_id) continue;

            float iou = calculateIoU(filtered_detections[i], filtered_detections[j]);
            if (iou >= iou_threshold) {
                cluster.push_back(j);
                is_clustered[j] = true;
            }
        }

        clusters.push_back(cluster);
    }

    // Fuse boxes in each cluster
    std::vector<Detection> fused_detections;
    for (const auto& cluster : clusters) {
        float total_score = 0.0f;
        float weighted_x1 = 0.0f, weighted_y1 = 0.0f, weighted_x2 = 0.0f, weighted_y2 = 0.0f;
        int max_class_id = filtered_detections[cluster[0]].class_id;

        for (int idx : cluster) {
            const auto& det = filtered_detections[idx];
            total_score += det.score;
            weighted_x1 += det.x1 * det.score;
            weighted_y1 += det.y1 * det.score;
            weighted_x2 += det.x2 * det.score;
            weighted_y2 += det.y2 * det.score;
        }

        // Calculate weighted average
        weighted_x1 /= total_score;
        weighted_y1 /= total_score;
        weighted_x2 /= total_score;
        weighted_y2 /= total_score;

        // Average score as the final score
        float avg_score = total_score / cluster.size();

        fused_detections.emplace_back(
            weighted_x1, weighted_y1, weighted_x2, weighted_y2,
            avg_score, max_class_id
        );
    }

    return fused_detections;
}

// Rest of the code remains the same...

// Main function and other functions remain the same as in your original code
