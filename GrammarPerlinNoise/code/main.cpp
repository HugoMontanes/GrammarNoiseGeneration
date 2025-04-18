#include "Scene.hpp"
#include "GeneticAlgorithm.hpp"
#include "Window.hpp"
#include "iostream"
#include <thread>

using space::Window;

int main(int, char* []) {
    constexpr unsigned viewport_width = 256;
    constexpr unsigned viewport_height = 256;

    Window window("GrammarNoise", Window::Position::CENTERED, Window::Position::CENTERED,
        viewport_width, viewport_height, { 3,3 });

    space::Scene scene(viewport_width, viewport_height);

    // Create and run the genetic algorithm
    space::GeneticAlgorithm ga(&scene, 50, 0.8f, 0.2f, 20);

    scene.configureScreenshotPath("../../../assets/generated_images");

    // Set parameter constraints
    ga.setParameterConstraints(1.0f, 10.0f, 0.1f, 1.0f, 1, 5);

    // Run optimization in a separate thread
    std::thread gaThread([&ga]() {
        ga.run();
        });

    bool running = true;
    SDL_Event event;

    Uint64 NOW = SDL_GetPerformanceCounter();
    Uint64 LAST = 0;
    double deltaTime = 0;

    // Print controls
    std::cout << "Controls:" << std::endl;
    std::cout << "Z - Take screenshot" << std::endl;
    std::cout << "G - Start/Stop genetic algorithm" << std::endl;

    bool gaRunning = true;

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
                ga.stop();
            }
            else if (event.type == SDL_WINDOWEVENT) {
                if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
                    scene.resize(viewport_width, viewport_height);
                }
            }
            else if (event.type == SDL_KEYDOWN) {
                // Add a dedicated key for screenshots
                if (event.key.keysym.scancode == SDL_SCANCODE_P) {
                    scene.takeScreenshot(space::ScreenshotExporter::ImageFormat::PNG);
                }
                // Toggle GA on/off
                if (event.key.keysym.scancode == SDL_SCANCODE_G) {
                    if (gaRunning) {
                        ga.stop();
                        gaRunning = false;
                        std::cout << "Genetic algorithm stopped." << std::endl;
                    }
                    else {
                        gaThread = std::thread([&ga]() {
                            ga.run();
                            });
                        gaRunning = true;
                        std::cout << "Genetic algorithm started." << std::endl;
                    }
                }
                // Apply current best solution
                if (event.key.keysym.scancode == SDL_SCANCODE_B) {
                    auto best = ga.getBestSolution();
                    scene.setFrequency(best.frequency);
                    scene.setAmplitude(best.amplitude);
                    scene.setOctaves(best.octaves);
                    std::cout << "Applied best solution: f=" << best.frequency
                        << ", a=" << best.amplitude << ", o=" << best.octaves << std::endl;
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

    // Wait for GA thread to finish
    if (gaThread.joinable()) {
        gaThread.join();
    }

    return 0;
}




