#pragma once

#include <string>
#include <glad/glad.h>

namespace space
{
	class Shader
	{
	protected:

		GLuint shader_id;
		GLenum shader_type;

	public:

		Shader(GLenum _type) : shader_id(0), shader_type(_type) {}

		virtual ~Shader()
		{
			if (shader_id)
			{
				glDeleteShader(shader_id);
			}
		}

		bool loadFromFile(const std::string& file_path);

		bool loadFromString(const std::string& source)
		{
			return compile(source);
		}

		GLuint getShaderID() const {
			return shader_id;
		}

	protected:

		bool compile(const std::string& source);
	};
}