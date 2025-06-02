/*
* Codigo realizado por Hugo Montañés García.
*/

#include "ONNXModel.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <locale>
#include <codecvt>
#include <Windows.h>
#include <algorithm>

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

    // Resize to model expected input size (256x256)
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(256, 256));

    // Convert to grayscale if needed (since model expects 256 channels)
    cv::Mat grayImage;
    if (resizedImage.channels() == 3 || resizedImage.channels() == 4) {
        cv::cvtColor(resizedImage, grayImage, cv::COLOR_BGR2GRAY);
    }
    else {
        grayImage = resizedImage;
    }

    // Convert to float, scale to [0,1]
    cv::Mat floatImage;
    grayImage.convertTo(floatImage, CV_32F, 1.0 / 255.0);

    // Reshape to match the expected dimensions (256x256x1)
    std::vector<float> inputTensor(256 * 256);

    // Copy data from OpenCV Mat to flat vector
    for (int h = 0; h < 256; h++) {
        for (int w = 0; w < 256; w++) {
            inputTensor[h * 256 + w] = floatImage.at<float>(h, w);
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

        // Get input node info to understand what the model expects
        Ort::TypeInfo typeInfo = session->GetInputTypeInfo(0);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> expectedDims = tensorInfo.GetShape();

        // Output expected dimensions for debugging
        std::cout << "Expected input dimensions: [";
        for (size_t i = 0; i < expectedDims.size(); i++) {
            std::cout << expectedDims[i];
            if (i < expectedDims.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::vector<int64_t> inputDims = expectedDims;
        for (auto& dim : inputDims) {
            if (dim < 0) dim = 1;  // Replace dynamic dimensions with batch size 1
        }

        // Calculate expected size based on the adjusted dimensions
        size_t expectedSize = 1;
        for (auto dim : inputDims) {
            expectedSize *= dim;  // All dimensions should be positive now
        }
        std::cout << "Expected input size: " << expectedSize << std::endl;
        std::cout << "Provided input size: " << inputTensor.size() << std::endl;

        // Define input shape
        //std::vector<int64_t> inputShape = { 1, 4, 256, 356 }; // Batch, Channels, Height, Width

        std::vector<float> resizedInputTensor;
        if (inputTensor.size() == expectedSize) {
            resizedInputTensor = std::move(inputTensor);
        }
        else {
            // Resize if necessary (we should avoid this situation by preprocessing correctly)
            resizedInputTensor.resize(expectedSize, 0.0f);
            size_t copySize = (std::min)(inputTensor.size(), expectedSize);
            std::copy_n(inputTensor.begin(), copySize, resizedInputTensor.begin());
            std::cout << "Warning: Input tensor size mismatch. Resized from "
                << inputTensor.size() << " to " << expectedSize << std::endl;
        }

        Ort::Value inputOnnxTensor = Ort::Value::CreateTensor<float>(
            *memoryInfo,
            resizedInputTensor.data(),
            resizedInputTensor.size(),
            inputDims.data(),
            inputDims.size()
        );

        // Create input tensor
        /*Ort::Value inputOnnxTensor = Ort::Value::CreateTensor<float>(
            *memoryInfo,
            inputTensor.data(),
            inputTensor.size(),
            inputShape.data(),
            inputShape.size()
        );*/

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

        // Print the result for debugging
        std::cout << "Fitness score: " << fitnessScore << std::endl;

        // Free allocated memory
        /*allocator.Free(const_cast<char*>(inputName));
        allocator.Free(const_cast<char*>(outputName));*/

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