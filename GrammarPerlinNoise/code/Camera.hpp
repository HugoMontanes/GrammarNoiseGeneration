
#pragma once

#include "glm.hpp"
#include <glad/glad.h>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>

#include <memory>
#include <string>
#include <vector>

#include "SceneNode.hpp"

namespace space
{
    class Camera : public SceneNode
    {
    public:

        float fov = 45.0f;
        float aspect = 1.0f;
        float nearPlane = 0.1f;
        float farPlane = 1000.0f;

        Camera(const std::string& name = "camera") : SceneNode(name) {};
            
        glm::mat4 getViewMatrix() const
        {
            return glm::inverse(getWorldTransform());
        }

        glm::mat4 getProjectionMatrix() const
        {
            return glm::perspective(glm::radians(fov), aspect, nearPlane, farPlane);
        }
        
    };
}

