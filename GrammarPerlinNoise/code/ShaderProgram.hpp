/*
* Codigo realizado por Hugo Montañés García.
*/

#pragma once

#include <glad/glad.h>
#include <iostream>
#include <vector>
#include <string>

#include "Shader.hpp"

namespace space
{
	class ShaderProgram
	{
	private:

		GLuint program_id;

	public:

		ShaderProgram() : program_id(glCreateProgram()) {}

		~ShaderProgram()
		{
			if (program_id)
			{
				glDeleteProgram(program_id);
			}
		}

		void attachShader(const Shader& shader) const
		{
			glAttachShader(program_id, shader.getShaderID());
		}

		bool link() const
		{
			glLinkProgram(program_id);

			/**
			* Check for linkage errors
			*/
			GLint success;
			glGetProgramiv(program_id, GL_LINK_STATUS, &success);
			if (!success) 
			{

				GLint log_length = 0;
				glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &log_length);

				std::vector<char> info_log(log_length);
				glGetProgramInfoLog(program_id, log_length, nullptr, info_log.data());

				std::cerr << "Shader program linking error:\n" << info_log.data() << std::endl;

				return false;
			}

			return true;

		}

		void use() const
		{
			glUseProgram(program_id);
		}

		GLuint getProgramID() const
		{
			return program_id;
		}

		void detachAndDeleteShaders(const std::vector<Shader>& shaders) const
		{
			for (const auto& shader : shaders)
			{
				glDetachShader(program_id, shader.getShaderID());
				glDeleteShader(shader.getShaderID());
			}
		}
	};
}