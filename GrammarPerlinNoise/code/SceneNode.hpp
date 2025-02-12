
#pragma once

#include"glm.hpp"
#include <glad/glad.h>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>

#include <memory>
#include <string>
#include <vector>

namespace space
{

    class SceneNode : public std::enable_shared_from_this<SceneNode>
    {
    public:

        std::string name;

        glm::vec3 position = glm::vec3(0.0f);
        glm::vec3 rotation = glm::vec3(0.0f);
        glm::vec3 scale = glm::vec3(1.0f);

        std::weak_ptr<SceneNode> parent;
        std::vector<std::shared_ptr<SceneNode>> children;

        std::shared_ptr<Mesh> mesh;

        SceneNode(const std::string& nodeName = "node") : name(nodeName){}

        void addChild(std::shared_ptr<SceneNode> child)
        {
            child->parent = shared_from_this();
            children.push_back(child);
        }

        glm::mat4 getLocalTransform() const
        {
            glm::mat4 transform(1.0f);

            // Apply transforms in order: Scale -> Rotate -> Translate
            transform = glm::translate(transform, position);
            transform = glm::rotate(transform, rotation.x, glm::vec3(1, 0, 0));
            transform = glm::rotate(transform, rotation.y, glm::vec3(0, 1, 0));
            transform = glm::rotate(transform, rotation.z, glm::vec3(0, 0, 1));
            transform = glm::scale(transform, scale);

            return transform;
        }

        glm::mat4 getWorldTransform() const
        {
            glm::mat4 transform = getLocalTransform();

            auto parentPtr = parent.lock();
            if (parentPtr)
            {
                transform = parentPtr->getWorldTransform() * transform;
            }

            return transform;
        }

    };
}
