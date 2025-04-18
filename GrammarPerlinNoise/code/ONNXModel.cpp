#include "ONNXModel.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <locale>
#include <codecvt>
#include <Windows.h>

ONNXModel::ONNXModel(const std::string& modelPath) : modelPath(modelPath) {
    try {
        // Initialize ONNX Runtime
        env = std::make_unique<Ort::Env>( ORT_LOGGING_LEVEL_WARNING, "VoronoiModel");

        // Session options
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Create session
        std::wstring wModelPath;
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, NULL, 0);
        wModelPath.resize(size_needed - 1);  // -1 because we don't need the null terminator in the string
        MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, &wModelPath[0], size_needed);
        session = std::make_unique<Ort::Session>(*env, wModelPath.c_str(), sessionOptions);

        // Create memory info
        memoryInfo = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

        std::cout << "ONNX model loaded successfully: " << modelPath << std::endl;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        throw;
    }
}

ONNXModel::~ONNXModel() {
    // Smart pointers handle cleanup
}

std::vector<float> ONNXModel::preprocessImage(const std::string& imagePath) {
    // Load image using OpenCV
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error loading image: " << imagePath << std::endl;
        return {};
    }

    // Resize to model expected input size (assuming 224x224)
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(224, 224));

    // Convert to float, scale to [0,1]
    cv::Mat floatImage;
    resizedImage.convertTo(floatImage, CV_32F, 1.0 / 255.0);

    // Normalize with ImageNet mean and std (if needed)
    // cv::Scalar mean(0.485, 0.456, 0.406);
    // cv::Scalar std(0.229, 0.224, 0.225);
    // floatImage = (floatImage - mean) / std;

    // Rearrange from HWC to CHW (height, width, channels) -> (channels, height, width)
    std::vector<float> inputTensor(3 * 224 * 224);

    // Copy data from OpenCV Mat to flat vector in CHW format
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < 224; h++) {
            for (int w = 0; w < 224; w++) {
                inputTensor[c * 224 * 224 + h * 224 + w] = floatImage.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    return inputTensor;
}

float ONNXModel::evaluateImage(const std::string& imagePath) {
    try 
    {
        // Preprocess image
        std::vector<float> inputTensor = preprocessImage(imagePath);
        if (inputTensor.empty()) {
            return 0.0f; // Return minimum fitness on error
        }

        // Define input shape
        std::vector<int64_t> inputShape = { 1, 3, 224, 224 }; // Batch, Channels, Height, Width

        // Create input tensor
        Ort::Value inputOnnxTensor = Ort::Value::CreateTensor<float>(
            *memoryInfo,
            inputTensor.data(),
            inputTensor.size(),
            inputShape.data(),
            inputShape.size()
        );

        // Define input and output names
        Ort::AllocatorWithDefaultOptions allocator;

        // Get input and output names
        Ort::AllocatedStringPtr inputNamePtr = session->GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr outputNamePtr = session->GetOutputNameAllocated(0, allocator);
        const char* inputName = inputNamePtr.get();
        const char* outputName = outputNamePtr.get();

        // Run inference
        std::vector<Ort::Value> outputTensors = session->Run(
            Ort::RunOptions{ nullptr },
            &inputName,
            &inputOnnxTensor,
            1,
            &outputName,
            1
        );

        // Get output data
        float* floatOutput = outputTensors[0].GetTensorMutableData<float>();

        // For a regression model, we expect a single output value
        // For a classification model, you might need to get the highest class probability
        float fitnessScore = floatOutput[0];

        // Free allocated memory
        allocator.Free(const_cast<char*>(inputName));
        allocator.Free(const_cast<char*>(outputName));

        return fitnessScore;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error during inference: " << e.what() << std::endl;
        return 0.0f; // Return minimum fitness on error
    }
    catch (const std::exception& e) {
        std::cerr << "Standard exception during inference: " << e.what() << std::endl;
        return 0.0f;
    }
}