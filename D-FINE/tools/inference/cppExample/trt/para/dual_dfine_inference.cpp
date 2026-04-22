#include <opencv2/opencv.hpp>
#include <NvInfer.h>
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

int getImageId(const std::string& img_name) {
    std::string base_name = img_name;

    // Remove .png or .jpg extension if present
    size_t ext_pos = base_name.find(".png");
    if (ext_pos != std::string::npos) {
        base_name = base_name.substr(0, ext_pos);
    } else {
        ext_pos = base_name.find(".jpg");
        if (ext_pos != std::string::npos) {
            base_name = base_name.substr(0, ext_pos);
        }
    }

    // Extract components
    size_t pos1 = base_name.find("camera");
    size_t pos2 = base_name.find("_", pos1);
    size_t pos3 = base_name.find("_", pos2 + 1);

    if (pos1 == std::string::npos || pos2 == std::string::npos || pos3 == std::string::npos) {
        return 0; // Default value if parsing fails
    }

    int camera_idx = std::stoi(base_name.substr(pos1 + 6, pos2 - (pos1 + 6)));

    char scene_char = base_name[pos2 + 1];
    int scene_idx = 0;
    if (scene_char == 'M') scene_idx = 0;
    else if (scene_char == 'A') scene_idx = 1;
    else if (scene_char == 'E') scene_idx = 2;
    else if (scene_char == 'N') scene_idx = 3;

    int frame_idx = std::stoi(base_name.substr(pos3 + 1));

    // Combine components to create image_id
    std::string id_str = std::to_string(camera_idx) + std::to_string(scene_idx) + std::to_string(frame_idx);
    return std::stoi(id_str);
}

// Function to preprocess an image for inference
void preprocessImage(const cv::Mat& img_bgr, float* input_data, int64_t* image_size_data, int target_size = 960) {
    // Store original dimensions
    image_size_data[0] = img_bgr.cols;
    image_size_data[1] = img_bgr.rows;

    // Use optimized OpenCV operations with pre-allocation
    static cv::Mat img_rgb;      // Static to avoid reallocation
    static cv::Mat img_resized;  // Static to avoid reallocation
    static cv::Mat img_float;    // Static to avoid reallocation

    // Convert BGR to RGB (efficient in-place operation if possible)
    cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);

    // Resize with INTER_NEAREST for speed (or INTER_LINEAR for quality)
    cv::resize(img_rgb, img_resized, cv::Size(target_size, target_size), 0, 0, cv::INTER_NEAREST);

    // Convert to float and normalize in one step
    img_resized.convertTo(img_float, CV_32FC3, 1.0/255.0);

    // Efficiently copy from HWC to CHW format with optimized memory access patterns
    const size_t step = img_float.step[0];
    const size_t pixel_size = img_float.elemSize1() * 3;
    const float* img_data = img_float.ptr<float>(0);

    // Use OpenMP to parallelize the channel transpose operation
    #pragma omp parallel for
    for (int c = 0; c < 3; c++) {
        float* channel_data = input_data + c * target_size * target_size;
        for (int h = 0; h < target_size; h++) {
            for (int w = 0; w < target_size; w++) {
                channel_data[h * target_size + w] = img_data[h * step/sizeof(float) + w * 3 + c];
            }
        }
    }
}

// Function to convert detections to JSON format for AI CITY submission
Json::Value detectionsToJson(const std::vector<Detection>& detections, const std::string& image_name) {
    Json::Value json_detections = Json::arrayValue;
    int image_id = getImageId(image_name);

    for (const auto& det : detections) {
        Json::Value detection;
        detection["image_id"] = image_id;
        detection["category_id"] = det.class_id;

        // Format as [x, y, width, height]
        Json::Value bbox = Json::arrayValue;
        bbox.append(det.x1);
        bbox.append(det.y1);
        bbox.append(det.x2 - det.x1);  // width
        bbox.append(det.y2 - det.y1);  // height

        detection["bbox"] = bbox;
        detection["score"] = det.score;

        json_detections.append(detection);
    }

    return json_detections;
}

// Print a progress bar
void printProgressBar(int current, int total, int bar_width = 50) {
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(bar_width * progress);

    std::cout << "[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% (" << current << "/" << total << ")\r";
    std::cout.flush();

    if (current == total) {
        std::cout << std::endl;
    }
}

// Warm up GPU to ensure consistent timing
void warmupGPU(TRTEngine& engine1, TRTEngine& engine2, int iterations = 50) {
    std::cout << "Warming up GPU with " << iterations << " iterations..." << std::endl;

    // Create dummy data
    const int input_size = 3 * 960 * 960;
    float* dummy_input = new float[input_size];
    int64_t dummy_size[2] = {960, 960};

    // Fill with random data
    for (int i = 0; i < input_size; i++) {
        dummy_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Perform warmup iterations
    for (int i = 0; i < iterations; i++) {
        engine1.infer(dummy_input, dummy_size);
        engine2.infer(dummy_input, dummy_size);

        if (i % 10 == 0) {
            printProgressBar(i, iterations);
        }
    }

    printProgressBar(iterations, iterations);
    cudaDeviceSynchronize();

    // Clean up
    delete[] dummy_input;
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    std::cout << "GPU warmup completed!" << std::endl;
}

int main(int argc, char* argv[]) {
    // Parse command-line arguments
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <engine1_path> <engine2_path> <input_folder> <output_json> [score_threshold] [iou_threshold] [skip_warmup]" << std::endl;
        return 1;
    }

    std::string engine1_path = argv[1];
    std::string engine2_path = argv[2];
    std::string input_folder = argv[3];
    std::string output_file = argv[4];
    float score_threshold = (argc > 5) ? std::stof(argv[5]) : 0.0001f;
    float iou_threshold = (argc > 6) ? std::stof(argv[6]) : 0.55f;
    bool skip_warmup = (argc > 7) ? std::string(argv[7]) == "1" : false;

    // Create CUDA streams for parallel execution
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    try {
        std::cout << "Loading TensorRT engines..." << std::endl;
        TRTEngine engine1(engine1_path, stream1);
        TRTEngine engine2(engine2_path, stream2);
        std::cout << "Engines loaded successfully!" << std::endl;

        // Warm up GPU if not skipped
        if (!skip_warmup) {
            warmupGPU(engine1, engine2, 200);
        }

        // Collect image paths from input folder
        std::vector<std::string> image_paths;
        for (const auto& entry : fs::directory_iterator(input_folder)) {
            std::string ext = entry.path().extension().string();
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                image_paths.push_back(entry.path().string());
            }
        }
        std::sort(image_paths.begin(), image_paths.end());

        std::cout << "Found " << image_paths.size() << " images to process" << std::endl;

        // Allocate memory for input data
        const int input_size = 3 * 960 * 960;
        float* input_data = new float[input_size];
        int64_t image_size_data[2];

        // Performance tracking
        Timer timer;
        double total_time = 0.0;
        double total_preprocess_time = 0.0;
        double total_inference_time = 0.0;
        double total_postprocess_time = 0.0;

        // Output JSON data
        Json::Value all_detections = Json::arrayValue;

        // Process each image
        for (size_t i = 0; i < image_paths.size(); i++) {
            const std::string& image_path = image_paths[i];
            std::string image_name = fs::path(image_path).filename().string();

            // Load image
            cv::Mat img_bgr = cv::imread(image_path);
            if (img_bgr.empty()) {
                std::cerr << "Failed to load image: " << image_path << std::endl;
                continue;
            }

            // Measure preprocessing time
            timer.start();
            preprocessImage(img_bgr, input_data, image_size_data);
            double preprocess_time = timer.stop();
            total_preprocess_time += preprocess_time;

            // Run inference on both models concurrently
            timer.start();
            engine1.infer(input_data, image_size_data);
            engine2.infer(input_data, image_size_data);

            // Synchronize both streams
            cudaStreamSynchronize(stream1);
            cudaStreamSynchronize(stream2);
            double inference_time = timer.stop();
            total_inference_time += inference_time;

            // Get detections from both models
            timer.start();
            auto detections1 = engine1.getDetections(score_threshold);
            auto detections2 = engine2.getDetections(score_threshold);

            // Apply Weighted Box Fusion
            auto fused_detections = applyWBF(detections1, detections2, iou_threshold, score_threshold);

            // Convert to JSON format
            Json::Value json_detections = detectionsToJson(fused_detections, image_name);
            for (int j = 0; j < json_detections.size(); j++) {
                all_detections.append(json_detections[j]);
            }

            double postprocess_time = timer.stop();
            total_postprocess_time += postprocess_time;

            total_time += (preprocess_time + inference_time + postprocess_time);

            // Show progress every 10 images
            if ((i + 1) % 10 == 0 || i == image_paths.size() - 1) {
                printProgressBar(i + 1, image_paths.size());
            }
        }

        // Write results to output file
        std::ofstream outfile(output_file);
        Json::StyledWriter writer;
        outfile << writer.write(all_detections);
        outfile.close();

        // Calculate and display performance metrics
        int processed_images = image_paths.size();
        double avg_preprocess_time = total_preprocess_time / processed_images;
        double avg_inference_time = total_inference_time / processed_images;
        double avg_postprocess_time = total_postprocess_time / processed_images;
        double avg_total_time = total_time / processed_images;
        double fps = processed_images / (total_time / 1000.0);
        double normalized_fps = std::min(fps / 25.0, 1.0);

        std::cout << "\n--- Performance Report ---" << std::endl;
        std::cout << "Processed images: " << processed_images << std::endl;
        std::cout << "Total time: " << total_time / 1000.0 << " seconds" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Avg preprocess time:   " << avg_preprocess_time << " ms" << std::endl;
        std::cout << "Avg inference time:    " << avg_inference_time << " ms" << std::endl;
        std::cout << "Avg postprocess time:  " << avg_postprocess_time << " ms" << std::endl;
        std::cout << "Avg total time:        " << avg_total_time << " ms" << std::endl;
        std::cout << "FPS: " << fps << std::endl;
        std::cout << "Normalized FPS: " << normalized_fps << std::endl;
        std::cout << "\nResults saved to: " << output_file << std::endl;
        std::cout << "Total detections: " << all_detections.size() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // Clean up
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}
