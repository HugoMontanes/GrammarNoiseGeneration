
#pragma once

#include <glad/glad.h>
#include <glm.hpp>
#include <vector>

#include <iostream>

namespace space
{
	class Mesh
	{
	protected:

		enum
		{
			VERTICES_VBO,
			NORMALS_VBO,
			COLORS_VBO,
			INDICES_EBO,
			VBO_COUNT
		};

		GLuint vbo_ids[VBO_COUNT];
		GLuint vao_id;

		std::vector < glm::vec3 > vertices;
		std::vector < glm::vec3 > normals;
		std::vector < glm::vec4 > colors;
		std::vector <GLuint> indices;

	public:

		/**
		* Makes shure all ids are initialized to 0.
		*/
		Mesh() : vbo_ids{ 0 }, vao_id(0)
		{
		}

		virtual ~Mesh() { cleanUp(); }

		/**
		* Forces the children to implement and initialize the vertices, normals, color and indices.
		*/
		virtual void initialize() = 0;

		void setUpMesh();

		virtual void render() 
		{
			if (vao_id == 0) 
			{ 
				std::cerr << "Error: VAO not initialized." << std::endl; 
				return; 
			}

			glBindVertexArray(vao_id);
			glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, nullptr);
			glBindVertexArray(0);

			GLenum error = glGetError(); 
			if (error != GL_NO_ERROR) 
			{ 
				std::cerr << "OpenGL error in render: " << error << std::endl; 
			}

		}

		void cleanUp()
		{
			glDeleteVertexArrays(1, &vao_id);
			glDeleteBuffers(VBO_COUNT, vbo_ids);
		}

	};
}