
#include "ScreenQuad.hpp"

namespace space
{
    void ScreenQuad::initialize()
    {
        //Create a quad that fills the entire screen.

        vertices =
        {
            glm::vec3(-1.0f, -1.0f, 0.0f),  // bottom left
            glm::vec3(1.0f, -1.0f, 0.0f),   // bottom right
            glm::vec3(1.0f, 1.0f, 0.0f),    // top right
            glm::vec3(-1.0f, 1.0f, 0.0f)    // top left
        };

        normals =
        {
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f)
        };

        colors =
        {
            glm::vec4(1.0f, 1.0f, 1.0f, 0.0f),  // bottom left
            glm::vec4(1.0f, 1.0f, 1.0f, 0.0f),  // bottom right
            glm::vec4(1.0f, 1.0f, 1.0f, 0.0f),  // top right
            glm::vec4(1.0f, 1.0f, 1.0f, 0.0f)   // top left
        };

        indices =
        {
            0, 1, 2,
            0, 2, 3
        };
    }
}