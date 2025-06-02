/*
* Codigo realizado por Hugo Montañés García.
*/

#include "Mesh.hpp"

namespace space
{
	void Mesh::setUpMesh()
	{

		/**
		* Generate IDs for the VAO and VBOs.
		*/
		glGenBuffers(VBO_COUNT, vbo_ids);
		glGenVertexArrays(1, &vao_id);

		/**
		* Binding the vao array.
		*/
		glBindVertexArray(vao_id);

		/**
		* Vertices vbo
		*/
		glBindBuffer(GL_ARRAY_BUFFER, vbo_ids[VERTICES_VBO]);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), vertices.data(), GL_STATIC_DRAW);
		
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);

		/**
		* Normals vbo
		*/
		glBindBuffer(GL_ARRAY_BUFFER, vbo_ids[NORMALS_VBO]);
		glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(glm::vec3), normals.data(), GL_STATIC_DRAW);
		
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);

		/**
		* Colors vbo
		*/
		glBindBuffer(GL_ARRAY_BUFFER, vbo_ids[COLORS_VBO]);
		glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec4), colors.data(), GL_STATIC_DRAW);
		
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), nullptr);

		/**
		* Indices ebo
		*/
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_ids[INDICES_EBO]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);

		/**
		* Unbind vao
		*/
		glBindVertexArray(0);

		GLenum error = glGetError(); 
		
		if (error != GL_NO_ERROR) 
		{ 
			std::cerr << "OpenGL error in setupMesh: " << error << std::endl; 
		}
	}
}