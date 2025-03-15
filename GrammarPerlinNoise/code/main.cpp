#include "Scene.hpp"
#include "Window.hpp"
#include "iostream"

using space::Window;

int main(int, char* [])
{

    constexpr unsigned viewport_width = 256;
    constexpr unsigned viewport_height = 256;

    Window window("GrammarNoise", Window::Position::CENTERED, Window::Position::CENTERED, viewport_width, viewport_height, { 3,3 });

    space::Scene scene(viewport_width, viewport_height);

    bool running = true;
    SDL_Event event;

    Uint64 NOW = SDL_GetPerformanceCounter();
    Uint64 LAST = 0;
    double deltaTime = 0;

    std::cout << "Controls:" << std::endl;
    std::cout << "WASD - Move camera" << std::endl;
    std::cout << "Arrow keys - Rotate camera" << std::endl;
    std::cout << "C - Reset camera rotation" << std::endl;
    std::cout << "Z - Take screenshot" << std::endl;

    while (running)
    {
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
            {
                running = false;
            }
            else if (event.type == SDL_WINDOWEVENT)
            {
                if (event.window.event == SDL_WINDOWEVENT_RESIZED)
                {
                    scene.resize(viewport_width, viewport_height);

                }
            }
            else if (event.type == SDL_KEYDOWN)
            {
                // Add a dedicated key for screenshots
                if (event.key.keysym.scancode == SDL_SCANCODE_F12)
                {
                    scene.takeScreenshot(space::ScreenshotExporter::ImageFormat::PNG);
                }
            }
        }
        LAST = NOW;
        NOW = SDL_GetPerformanceCounter();
        deltaTime = (double)((NOW - LAST) / (double)SDL_GetPerformanceFrequency());

        scene.update(deltaTime);
        scene.render();
        window.swap_buffers();
    }

    return 0;
}




