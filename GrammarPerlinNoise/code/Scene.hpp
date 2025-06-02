/*
* Codigo realizado por Hugo Montañés García.
*/

#pragma once

#include"glm.hpp"
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include <SDL.h>

#include <memory>
#include <string>
#include <vector>

#include "Mesh.hpp"
#include "ShaderProgram.hpp"
#include "VertexShader.hpp"
#include "FragmentShader.hpp"
#include "SceneNode.hpp"
#include "Camera.hpp"
#include "Shader.hpp"
#include "Screenshot.hpp"
#include "ScreenQuad.hpp"
#include "ShaderParameterLogger.hpp"
#include <unordered_map>

namespace space
{

    class Scene
    {
        SDL_Window* window = nullptr;
        std::unique_ptr<ShaderProgram> shader_program;
        std::shared_ptr<SceneNode> root;
        std::shared_ptr<Camera> activeCamera;
        std::shared_ptr<ScreenshotExporter> screenshotExporter;
        std::unique_ptr<ShaderParameterLogger> parameterLogger;

        glm::vec3 defaultCameraRotation = glm::vec3(0.0f); // Store default rotation
        std::unordered_map<SDL_Scancode, bool> keyStates;

        GLuint model_view_matrix_id = -1;
        GLuint projection_matrix_id = -1;
        GLint normal_matrix_id = -1;

        float angle;

        GLint noise_scale_id = -1;
        GLint time_id = -1;
        GLint frequency_id = -1;
        GLint amplitude_id = -1;
        GLint octaves_id = -1;

        // Current shader parameter values
        float currentFrequency = 2.5f;
        float currentAmplitude = 0.4f;
        int currentOctaves = 1;

        GLuint framebufferObject;
        GLuint frameTextureObject;
        GLuint depthRenderBuffer;
        unsigned fboWidth;
        unsigned int fboHeight;

        bool initFramebuffer(unsigned int width, unsigned int height);
        void deleteFramebuffer();
        bool resizeFramebuffer(unsigned int width, unsigned int height);


        bool saveFramebufferToFile(const std::string& filename, ScreenshotExporter::ImageFormat format);
        bool prepareFramebuffer();
        static const char* fbStatusString(GLenum status)
        {
            switch (status) {
            case GL_FRAMEBUFFER_COMPLETE:                      return "GL_FRAMEBUFFER_COMPLETE";
            case GL_FRAMEBUFFER_UNDEFINED:                     return "GL_FRAMEBUFFER_UNDEFINED";
            case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:         return "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT";
            case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: return "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";
            case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:        return "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER";
            case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:        return "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER";
            case GL_FRAMEBUFFER_UNSUPPORTED:                   return "GL_FRAMEBUFFER_UNSUPPORTED";
            case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:        return "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE";
            case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:      return "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS";
            default:                                           return "Unknown FBO status";
            }
        }

        bool exitRequested = false;

    public:
        
        Scene(unsigned width, unsigned height);
        ~Scene();

        void update(float deltaTime);
        void render();
        void renderToFBO();
        void resize(unsigned width, unsigned height);
        void renderNode(const std::shared_ptr<SceneNode>& node, const glm::mat4& viewMatrix);
        std::shared_ptr<SceneNode> createNode(const std::string& name, std::shared_ptr<SceneNode> parent = nullptr);
        std::shared_ptr<SceneNode> findNode(const std::string& name, const std::shared_ptr<SceneNode>& startNode);
        void handleKeyboard(const Uint8* keyboardState);

        bool takeScreenshot(ScreenshotExporter::ImageFormat format = ScreenshotExporter::ImageFormat::PNG);

        // Shader parameter controls
        void setFrequency(float value) {currentFrequency = value;}
        void setAmplitude(float value) {currentAmplitude = value;}
        void setOctaves(int value) {currentOctaves = value;}
        float getFrequency() const { return currentFrequency; }
        float getAmplitude() const { return currentAmplitude; }
        int getOctaves() const { return currentOctaves; }

        std::shared_ptr<ScreenshotExporter> getScreenshotExporter()
        {
            return screenshotExporter;
        }

        void configureScreenshotPath(const std::string& path)
        {
            screenshotExporter = std::make_shared<ScreenshotExporter>(path);
        }

        bool checkFramebufferStatus();

        bool isExitRequested()const { return exitRequested;  }
    };
}