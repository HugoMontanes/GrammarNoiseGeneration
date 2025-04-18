#pragma once

#include <string>
#include <vector>
#include <memory>

// Forward declarations
namespace Ort 
{
    class Session;
    struct Env;
    struct MemoryInfo;
}

class ONNXModel 
{
private:
    std::string modelPath;
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::MemoryInfo> memoryInfo;

    std::vector<float> preprocessImage(const std::string& imagePath);

public:
    ONNXModel(const std::string& modelPath);
    ~ONNXModel();

    // Evaluate an image and return fitness score
    float evaluateImage(const std::string& imagePath);
};