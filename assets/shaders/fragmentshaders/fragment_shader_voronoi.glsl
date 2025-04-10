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
uniform float frequency;
uniform float amplitude;
uniform int octaves;

// Hash function for randomization
vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

// Hash function that returns RGB color
vec3 hash3(vec2 p) {
    vec3 h = vec3(dot(p, vec2(127.1, 311.7)), 
                  dot(p, vec2(269.5, 183.3)), 
                  dot(p, vec2(419.2, 371.9)));
    return fract(sin(h) * 43758.5453123);
}

// Structure to hold Voronoi cell information
struct VoronoiResult {
    float dist;  // Distance to closest point
    vec2 point;  // Closest point
    vec2 cell;   // Cell ID (grid coordinates)
};

// Voronoi noise implementation that returns additional information
VoronoiResult voronoiDetailed(vec2 p) {
    vec2 pi = floor(p);
    vec2 pf = fract(p);
    
    VoronoiResult result;
    result.dist = 8.0;  // Initialize with large value
    result.cell = vec2(0.0);
    result.point = vec2(0.0);
    
    // Check neighboring cells
    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(x, y);
            vec2 cellId = pi + neighbor;
            vec2 point = hash2(cellId);
            
            // Animate point position slightly with time
            point = 0.5 * sin(time * 0.3 + 6.2831 * point);
            
            vec2 diff = neighbor + point - pf;
            float dist = length(diff);
            
            if(dist < result.dist) {
                result.dist = dist;
                result.cell = cellId;
                result.point = point;
            }
        }
    }
    
    return result;
}

// FBM (Fractal Brownian Motion) for multiple octaves of Voronoi noise
VoronoiResult fbm(vec2 p) {
    VoronoiResult result = voronoiDetailed(p * frequency);
    float value = amplitude * result.dist;
    float freq = frequency;
    float amp = amplitude;
    
    // Add multiple layers of noise
    for(int i = 1; i < octaves; i++) {
        freq *= 2.0;
        amp *= 0.5;
        VoronoiResult octave = voronoiDetailed(p * freq);
        value += amp * octave.dist;
    }
    
    result.dist = value;
    return result;
}

void main() {
    // Use the screen-space coordinates for the noise
    vec2 noiseCoord = gl_FragCoord.xy * noise_scale / 100.0;
    
    // Add time for animation
    noiseCoord += time * 0.1;
    
    // Generate detailed Voronoi information
    VoronoiResult result = fbm(noiseCoord);
    
    // Generate color based on cell ID
    vec3 cellColor = hash3(result.cell);
    
    // Create edges between cells
    float edge = smoothstep(0.02, 0.05, result.dist);
    
    // Final color: bright cell colors with dark edges
    vec3 finalColor = mix(vec3(0.1), cellColor * 1.2, edge);
    
    // Output the final color
    fragment_color = vec4(finalColor, 1.0);
}