
#pragma once

#include"glm.hpp"
#include <glad/glad.h>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include <SDL.h>

#include <memory>
#include <string>
#include <vector>

#include "Mesh.hpp"
#include "Plane.hpp"
#include "ShaderProgram.hpp"
#include "VertexShader.hpp"
#include "FragmentShader.hpp"
#include "SceneNode.hpp"
#include "Camera.hpp"
#include "Screenshot.hpp"
#include "ScreenQuad.hpp"
#include "ShaderParameterLogger.hpp"
#include <unordered_map>

namespace space
{

    class Scene
    {

        std::unique_ptr<ShaderProgram> shader_program;
        std::shared_ptr<SceneNode> root;
        std::shared_ptr<Camera> activeCamera;
        std::unique_ptr<ScreenshotExporter> screenshotExporter;
        std::unique_ptr<ShaderParameterLogger> parameterLogger;

        float cameraSpeed = 10.0f;
        float cameraRotationSpeed = 2.0f;
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

    public:
        
        Scene(unsigned width, unsigned height);

        void update(float deltaTime);
        void render();
        void resize(unsigned width, unsigned height);
        void renderNode(const std::shared_ptr<SceneNode>& node, const glm::mat4& viewMatrix);
        std::shared_ptr<SceneNode> createNode(const std::string& name, std::shared_ptr<SceneNode> parent = nullptr);
        std::shared_ptr<SceneNode> findNode(const std::string& name, const std::shared_ptr<SceneNode>& startNode);
        void handleKeyboard(const Uint8* keyboardState);
        void updateCamera(float deltaTime);
        void resetCameraRotation();

        bool takeScreenshot(ScreenshotExporter::ImageFormat format = ScreenshotExporter::ImageFormat::PNG);

        // Shader parameter controls
        void setFrequency(float value);
        void setAmplitude(float value);
        void setOctaves(int value);
    };
}