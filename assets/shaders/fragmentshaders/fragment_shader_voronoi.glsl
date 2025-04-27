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
    return fract(sin(p) * 43758.5453123) * 2.0 - 1.0;
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
    float dist;      // Distance to edge (using Inigo's method)
    vec2 cell;       // Closest cell ID for coloring
    vec2 point;      // Closest point
};

// Enhanced Voronoi implementation using Inigo's edge detection approach
VoronoiResult voronoiEnhanced(vec2 p) {
    vec2 pi = floor(p);
    vec2 pf = fract(p);
    
    // First pass: find closest point
    float minDist = 8.0;
    vec2 minPoint = vec2(0.0);
    vec2 minCell = vec2(0.0);
    vec2 minOffset = vec2(0.0);
    
    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(x, y);
            vec2 cellId = pi + neighbor;
            
            // Generate point within cell
            vec2 cellPoint = hash2(cellId) * 0.5;
            vec2 diff = neighbor + cellPoint - pf;
            float dist = dot(diff, diff); // Squared distance for speed
            
            if(dist < minDist) {
                minDist = dist;
                minPoint = cellPoint;
                minCell = cellId;
                minOffset = diff;
            }
        }
    }
    
    // Second pass: find edge distance using Inigo's method
    float edgeDist = 8.0;
    
    for(int y = -2; y <= 2; y++) {
        for(int x = -2; x <= 2; x++) {
            vec2 neighbor = vec2(x, y);
            vec2 cellId = pi + neighbor;
            
            // Skip the closest cell (already processed)
            if(cellId == minCell) continue;
            
            // Generate point within cell
            vec2 cellPoint = hash2(cellId) * 0.5;
            vec2 diff = neighbor + cellPoint - pf;
            
            // Calculate perpendicular distance to the edge
            // This is the key part of Inigo's approach
            float dist = dot(0.5 * (minOffset + diff), normalize(diff - minOffset));
            edgeDist = min(edgeDist, dist);
        }
    }
    
    VoronoiResult result;
    result.dist = edgeDist;
    result.cell = minCell;
    result.point = minPoint;
    
    return result;
}

// FBM (Fractal Brownian Motion) for multiple octaves
VoronoiResult fbmVoronoi(vec2 p) {
    // Start with base octave
    VoronoiResult result = voronoiEnhanced(p * frequency);
    
    // Store base values
    float value = amplitude * result.dist;
    vec2 baseCell = result.cell;
    
    // Accumulate additional octaves
    float freq = frequency;
    float amp = amplitude;
    
    for(int i = 1; i < octaves; i++) {
        freq *= 2.0;
        amp *= 0.5;
        
        // Get higher frequency detail
        VoronoiResult octave = voronoiEnhanced(p * freq);
        
        // Add detail to distance values
        value += amp * octave.dist;
    }
    
    // Apply accumulated distances but keep original cell info for consistent coloring
    result.dist = value;
    result.cell = baseCell;
    return result;
}

void main() {
    // Use the screen-space coordinates for the noise
    vec2 noiseCoord = gl_FragCoord.xy * noise_scale / 100.0;
    
    // Generate detailed Voronoi information
    VoronoiResult result = fbmVoronoi(noiseCoord);
    
    // Generate color based on cell ID
    vec3 cellColor = hash3(result.cell);
    
    // Apply smoothstep for controllable edge thickness
    float edgeThreshold = 0.05 * (1.0/frequency); 
    float edgeFactor = smoothstep(0.0, edgeThreshold, result.dist);
    
    // Final color: cell colors with transparent edges
    // edgeFactor = 1.0 means fully opaque (no edge)
    // edgeFactor = 0.0 means fully transparent (edge)
    
    // Output the final color with alpha channel
    // The edges will be invisible (transparent) because edgeFactor will be 0 at edges
    fragment_color = vec4(cellColor, edgeFactor);
}