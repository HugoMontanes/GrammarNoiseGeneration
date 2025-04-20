#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 color;

out vec3 vertex_position;
out vec3 vertex_normal;
out vec4 vertex_color;

uniform mat4 model_view_matrix;
uniform mat4 projection_matrix;
uniform mat4 normal_matrix;

void main() {
    vertex_position = position;
    vertex_normal = normal;
    vertex_color = color;
    
    gl_Position = projection_matrix * model_view_matrix * vec4(position, 1.0);
}