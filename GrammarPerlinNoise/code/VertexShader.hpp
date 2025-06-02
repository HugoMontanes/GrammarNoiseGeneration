/*
* Codigo realizado por Hugo Montañés García.
*/

#pragma once

#include "Shader.hpp"

namespace space
{
	class VertexShader : public Shader
	{

	public:

		VertexShader() : Shader(GL_VERTEX_SHADER) {}

	};
}