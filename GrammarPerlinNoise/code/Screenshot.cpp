/*
* Codigo realizado por Hugo Monta��s Garc�a.
*/

#include "Screenshot.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace space
{
    ScreenshotExporter::ScreenshotExporter(const std::string& path)
    {
        outputPath = path;
        // Ensure outputPath ends with a slash
        if (!outputPath.empty() && outputPath.back() != '/' && outputPath.back() != '\\') {
            outputPath += '/';
        }

        // Try to create the directory
        ensureDirectoryExists(outputPath);
    }

    bool ScreenshotExporter::captureScreenshot(unsigned int width, unsigned int height, ImageFormat format)
    {
        // Check if we're in a valid OpenGL context
        if (!gladLoadGL()) {
            std::cerr << "Error: Trying to capture screenshot without valid OpenGL context" << std::endl;
            return false;
        }

        // Get actual viewport dimensions if not specified
        if (width == 0 || height == 0) {
            GLint viewport[4];
            glGetIntegerv(GL_VIEWPORT, viewport);
            width = viewport[2];
            height = viewport[3];
        }

        // Print dimensions for debugging
        std::cout << "Capturing screenshot with dimensions: " << width << "x" << height << std::endl;

        // Allocate memory for the pixel data
        std::vector<unsigned char> pixels(width * height * 4);

        // Make sure we're reading from the correct framebuffer
        GLint currentFBO;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &currentFBO);
        std::cout << "Current framebuffer: " << currentFBO << std::endl;

        // Clear any previous errors
        while (glGetError() != GL_NO_ERROR);

        // Read pixels from the framebuffer
        glPixelStorei(GL_PACK_ALIGNMENT, 1); // Important for correct data alignment
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

        // Check for OpenGL errors
        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            std::cerr << "OpenGL error in captureScreenshot: " << error << std::endl;
            return false;
        }

        // Create filename with sequential numbering
        std::ostringstream filename;
        filename << outputPath << "image_" << counter++;

        // Add appropriate file extension
        switch (format) {
        case ImageFormat::PNG:
            filename << ".png";
            break;
        case ImageFormat::JPG:
            filename << ".jpg";
            break;
        }

        std::cout << "Saving screenshot to: " << filename.str() << std::endl;

        // Save the image
        return saveImage(filename.str(), width, height, pixels, format);
    }

    bool ScreenshotExporter::saveImage(const std::string& filename, unsigned int width, unsigned int height, const std::vector<unsigned char>& pixels, ImageFormat format)
    {

        // Flip the image vertically as OpenGL's origin is bottom left
        std::vector<unsigned char> flippedPixels(width * height * 4);
        for (unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                for (unsigned int c = 0; c < 4; ++c) {  
                    flippedPixels[(y * width + x) * 4 + c] =
                        pixels[((height - 1 - y) * width + x) * 4 + c];
                }
            }
        }

        int success = 0;

        // Write directly to file using stb functions
        switch (format) {
        case ImageFormat::PNG:
            // Use stride parameter correctly - width * 3 for RGB data
            success = stbi_write_png(filename.c_str(), width, height, 4, flippedPixels.data(), width * 4);
            break;
        case ImageFormat::JPG:
            success = stbi_write_jpg(filename.c_str(), width, height, 3, flippedPixels.data(), 90); // Quality 90
            break;
        }

        if (success) {
            std::cout << "Screenshot saved to: " << filename << std::endl;
            return true;
        }
        else {
            std::cerr << "Failed to save screenshot to: " << filename << std::endl;
            return false;
        }
    }

    
    bool ScreenshotExporter::ensureDirectoryExists(const std::string& dirPath) {
    // Create directories recursively
    std::string currentPath;
    for (char c : dirPath) {
        currentPath += c;
        if ((c == '/' || c == '\\') && !currentPath.empty()) {
#ifdef _WIN32
            if (_mkdir(currentPath.c_str()) != 0 && errno != EEXIST) {
                return false;
            }
#else
            if (mkdir(currentPath.c_str(), 0777) != 0 && errno != EEXIST) {
                return false;
            }
#endif
        }
    }
    return true;
}
}

