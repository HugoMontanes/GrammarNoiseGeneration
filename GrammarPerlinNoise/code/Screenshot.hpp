
#pragma once

#include <glad/glad.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <errno.h>
#ifdef _WIN32
#include <direct.h>
#endif



namespace space
{
    class ScreenshotExporter
    {
    private:
        int counter = 1;
        std::string outputPath = "./";
    public:
        enum class ImageFormat
        {
            PNG,
            JPG
        };

        ScreenshotExporter(const std::string& path);

        bool captureScreenshot(unsigned int width, unsigned int height, ImageFormat format = ImageFormat::PNG);

        // Add getter for the last image counter
        int getLastImageCounter() const { return counter - 1; }
        std::string getOutputPath() const { return outputPath; }


    private:

        bool saveImage(const std::string& filename, unsigned int width, unsigned int height, const std::vector<unsigned char>& pixels, ImageFormat format);
        bool ensureDirectoryExists(const std::string& path);
    };
}