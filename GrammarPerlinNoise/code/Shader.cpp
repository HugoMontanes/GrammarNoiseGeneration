/*
* Codigo realizado por Hugo Montañés García.
*/

#include"Shader.hpp"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>

namespace space
{
	bool Shader::loadFromFile(const std::string& file_path)
	{
		std::fstream file(file_path);
		if (!file.is_open())
		{
			std::cerr << "Failed to open shader file: " << file_path << std::endl; 
			return false;
		}

		std::stringstream buffer;
		buffer << file.rdbuf();
		file.close();

		return compile(buffer.str());
	}

	bool Shader::compile(const std::string& source)
	{
		shader_id = glCreateShader(shader_type);
		const char* source_cstr[]{ source.c_str() };	///< Transform to c_str because openGL is written in C not C++
		const GLint source_size[]{ GLint(source.size()) };
		glShaderSource(shader_id, 1, source_cstr, source_size);
		glCompileShader(shader_id);

		/**
		* Checks for compilation errors.
		*/
		GLint success;
		glGetShaderiv(shader_id, GL_COMPILE_STATUS, &success);
		if (!success) {
			char info_log[512];
			glGetShaderInfoLog(shader_id, 512, nullptr, info_log);
			std::cerr << "Shader compilation error:\n" << info_log << std::endl;
			return false;
		}

		return true;
	}
}