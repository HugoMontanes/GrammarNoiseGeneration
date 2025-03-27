#version 330 core

in vec3 vertex_position;
in vec3 vertex_normal;
in vec3 vertex_color;

out vec4 fragment_color;

uniform mat4 model_view_matrix;
uniform mat4 projection_matrix;
uniform mat4 normal_matrix;
uniform float time;
uniform float noise_scale;

// Hash function for randomization
vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

// Voronoi noise implementation
float voronoi(vec2 p) {
    vec2 pi = floor(p);
    vec2 pf = fract(p);
    
    float minDist = 1.0;
    
    // Check neighboring cells
    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(x, y);
            vec2 point = hash2(pi + neighbor);
            
            // Animate point position slightly with time
            point = 0.5 * sin(time + 6.2831 * point);
            
            vec2 diff = neighbor + point - pf;
            float dist = length(diff);
            
            minDist = min(minDist, dist);
        }
    }
    
    return minDist;
}

// FBM (Fractal Brownian Motion) for multiple octaves of Voronoi noise
float fbm(vec2 p) {
    float value = 0.1;
    float amplitude = 0.4;
    float frequency = 2.5;
    int octaves = 1;
    
    // Add multiple layers of noise
    for(int i = 0; i < octaves; i++) {
        value += amplitude * voronoi(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    
    return value;
}

void main() {
    // Use the screen-space coordinates for the noise
    vec2 noiseCoord = gl_FragCoord.xy * noise_scale / 100.0;
    
    // Add time for animation
    noiseCoord += time * 0.5;
    
    // Generate base noise value
    float noise = fbm(noiseCoord);
    
    // Map noise from [0,1] range with contrast adjustment
    float grayscale = clamp(noise * 1.2 - 0.1, 0.0, 1.0);
    
    // Create grayscale color
    vec3 finalColor = vec3(grayscale);
    
    // Output the final color
    fragment_color = vec4(finalColor, 1.0);
}