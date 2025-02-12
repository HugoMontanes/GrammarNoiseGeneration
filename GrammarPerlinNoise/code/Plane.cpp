

#include "Plane.hpp"
#include <vector>

namespace space
{
	void Plane::initialize()
	{
		float cell_width = width / float(cols);
		float cell_height = height / float(rows);

		vertices.reserve((cols + 1) * (rows + 1));
		normals.reserve((cols + 1) * (rows + 1));
		colors.reserve((cols + 1) * (rows + 1));
		indices.reserve(rows * cols * 6);

		for (unsigned int row = 0; row <= rows; ++row)
		{
			for (unsigned int col = 0; col <= cols; ++col)
			{
				GLfloat x = col * cell_width - width / 2.0f;
				GLfloat z = row * cell_height - height / 2.0f;

				/**
				* Vertex position
				*/
				vertices.emplace_back(x, 0.0f, z);

				/**
				* Normals (pointing up for a flat plane)
				*/
				normals.emplace_back(0.0f, 1.0f, 0.0f);

				/**
				* Colors
				*/
				colors.emplace_back(1.0f, 1.0f, 1.0f);
			}
		}

		for (unsigned int row = 0; row < rows; ++row)
		{
			for (unsigned int col = 0; col < cols; ++col)
			{
				GLuint top_left = row * (cols + 1) + col;
				GLuint top_right = top_left + 1;
				GLuint bottom_left = (row + 1) * (cols + 1) + col;
				GLuint bottom_right = bottom_left + 1;

				indices.push_back(top_left);
				indices.push_back(bottom_left);
				indices.push_back(top_right);

				indices.push_back(top_right);
				indices.push_back(bottom_left);
				indices.push_back(bottom_right);
			}
		}
	}
}