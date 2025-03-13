#pragma once

#include "Mesh.hpp"

namespace space
{
    class ScreenQuad : public Mesh
    {
    public:

        ScreenQuad()
        {
            initialize();
            setUpMesh();
        }

        void initialize() override;
    };
}
