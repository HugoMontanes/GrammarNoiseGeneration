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
    p = vec2(dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)));
    return -1.0 + 2.0 * fract(sin(p)*43758.5453123);
}

// 2D Perlin noise implementation
float perlin2D(vec2 p) {
    vec2 pi = floor(p);
    vec2 pf = fract(p);
    
    vec2 w = pf * pf * (3.0 - 2.0 * pf); // Smoothstep
    
    float n00 = dot(hash2(pi + vec2(0.0, 0.0)), pf - vec2(0.0, 0.0));
    float n10 = dot(hash2(pi + vec2(1.0, 0.0)), pf - vec2(1.0, 0.0));
    float n01 = dot(hash2(pi + vec2(0.0, 1.0)), pf - vec2(0.0, 1.0));
    float n11 = dot(hash2(pi + vec2(1.0, 1.0)), pf - vec2(1.0, 1.0));
    
    return mix(
        mix(n00, n10, w.x),
        mix(n01, n11, w.x),
        w.y
    );
}

// FBM (Fractal Brownian Motion) for multiple octaves of noise
float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
	int octaves = 6;
    
    // Add multiple layers of noise
    for(int i = 0; i < octaves; i++) {
        value += amplitude * perlin2D(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    
    return value;
}

void main() {
    // Use the screen-space coordinates for the noise
    // This ensures the noise covers the entire screen
    vec2 noiseCoord = gl_FragCoord.xy * noise_scale / 100.0;
    
    // Add time for animation if desired
    noiseCoord += time * 0.1;
    
    // Generate base noise value
    float noise = fbm(noiseCoord);
    
    // Map noise from [-1,1] to [0,1] range with more contrast
    float grayscale = clamp(noise * 0.8 + 0.5, 0.0, 1.0);
    
    // Create grayscale color
    vec3 finalColor = vec3(grayscale);
    
    // Output the final color
    fragment_color = vec4(finalColor, 1.0);
}