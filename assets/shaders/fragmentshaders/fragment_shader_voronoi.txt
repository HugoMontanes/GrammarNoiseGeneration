#version 330 core

in vec3 vertex_position;
in vec3 vertex_normal;
in vec4 vertex_color;

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
    float dist1;     // Distance to closest point
    float dist2;     // Distance to second closest point
    vec2 point1;     // Closest point
    vec2 point2;     // Second closest point
    vec2 cell1;      // Closest cell ID
    vec2 cell2;      // Second closest cell ID
};

// Enhanced Voronoi implementation using closest and second-closest points
VoronoiResult voronoiEnhanced(vec2 p) {
    vec2 pi = floor(p);
    vec2 pf = fract(p);
    
    VoronoiResult result;
    result.dist1 = 8.0;  // Initialize with large values
    result.dist2 = 8.0;
    result.cell1 = vec2(0.0);
    result.cell2 = vec2(0.0);
    result.point1 = vec2(0.0);
    result.point2 = vec2(0.0);
    
    // Check neighboring cells (expanding search radius to 2 for better edge quality)
    for(int y = -2; y <= 2; y++) {
        for(int x = -2; x <= 2; x++) {
            vec2 neighbor = vec2(x, y);
            vec2 cellId = pi + neighbor;
            
            // Generate point within cell - STATIC version (no time)
            vec2 cellPoint = hash2(cellId);
            cellPoint = 0.5 * sin( 6.2831 * cellPoint);
            
            // Calculate distance to this point
            vec2 diff = neighbor + cellPoint - pf;
            float dist = length(diff);
            
            // Update closest and second closest distances
            if(dist < result.dist1) {
                result.dist2 = result.dist1;
                result.cell2 = result.cell1;
                result.point2 = result.point1;
                
                result.dist1 = dist;
                result.cell1 = cellId;
                result.point1 = cellPoint;
            } 
            else if(dist < result.dist2) {
                result.dist2 = dist;
                result.cell2 = cellId;
                result.point2 = cellPoint;
            }
        }
    }
    
    return result;
}

// Improved FBM (Fractal Brownian Motion) for multiple octaves of Voronoi noise
VoronoiResult fbmVoronoi(vec2 p) {
    // Start with base octave
    VoronoiResult result = voronoiEnhanced(p * frequency);
    
    // Store base values
    float value1 = amplitude * result.dist1;
    float value2 = amplitude * result.dist2;
    
    // Accumulate additional octaves
    float freq = frequency;
    float amp = amplitude;
    
    for(int i = 1; i < octaves; i++) {
        freq *= 2.0;
        amp *= 0.5;
        
        // Get higher frequency detail
        VoronoiResult octave = voronoiEnhanced(p * freq);
        
        // Add detail to distance values
        value1 += amp * octave.dist1;
        value2 += amp * octave.dist2;
    }
    
    // Apply accumulated distances but keep original cell info
    result.dist1 = value1;
    result.dist2 = value2;
    return result;
}

void main() {
    // Use the screen-space coordinates for the noise
    vec2 noiseCoord = gl_FragCoord.xy * noise_scale / 100.0;
    
    // Generate detailed Voronoi information
    VoronoiResult result = fbmVoronoi(noiseCoord);
    
    // Generate color based on cell ID
    vec3 cellColor = hash3(result.cell1);
    
    // Calculate the edge using the difference between first and second closest points
    float edge = result.dist2 - result.dist1;
    
    // Apply smoothstep for controllable edge thickness
    float edgeThreshold = 0.03 * (1.0/frequency) * amplitude; 
    float edgeThickness = 0.05 * (1.0/frequency);
    float edgeFactor = smoothstep(edgeThickness, edgeThreshold + edgeThickness, edge);
    
    // Final color: bright cell colors with dark edges
    vec3 finalColor = mix(vec3(0.05), cellColor, edgeFactor);
    
    // Output the final color
    fragment_color = vec4(finalColor, edgeFactor);
}