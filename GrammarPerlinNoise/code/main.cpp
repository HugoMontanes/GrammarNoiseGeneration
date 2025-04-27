#include "Scene.hpp"
#include "GeneticAlgorithm.hpp"
#include "Window.hpp"
#include "iostream"
#include <thread>

using space::Window;

int main(int, char* []) {
    constexpr unsigned viewport_width = 2048;
    constexpr unsigned viewport_height = 2048;

    Window window("GrammarNoise", Window::Position::CENTERED, Window::Position::CENTERED,
        viewport_width, viewport_height, { 3,3 });

    space::Scene scene(viewport_width, viewport_height);

    // Create GA but don't run it automatically
    space::GeneticAlgorithm ga(&scene, 10, 0.8f, 0.2f, 10);
    scene.configureScreenshotPath("../../../assets/generated_images");
    ga.setParameterConstraints(1.0f, 10.0f, 0.1f, 1.0f, 1, 5);
    

    // GA control variables
    bool gaRunning = false;
    bool gaInitialized = false;

    // Add a mode variable for UI
    enum class Mode { MANUAL, GENETIC_ALGORITHM };
    Mode currentMode = Mode::MANUAL;

    bool running = true;
    SDL_Event event;

    Uint64 NOW = SDL_GetPerformanceCounter();
    Uint64 LAST = 0;
    double deltaTime = 0;

    // Print controls
    std::cout << "Controls:" << std::endl;
    std::cout << "Z - Take screenshot" << std::endl;
    std::cout << "G - Start/Stop genetic algorithm" << std::endl;
    std::cout << "B - Apply best solution from GA" << std::endl;
    std::cout << "Arrow keys - Manual frequency adjustment (when GA inactive)" << std::endl;
    std::cout << "Shift+Arrow keys - Manual amplitude adjustment" << std::endl;
    std::cout << "Ctrl+Arrow keys - Manual octaves adjustment" << std::endl;

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
                // Screenshot key
                if (event.key.keysym.scancode == SDL_SCANCODE_Z) {
                    scene.takeScreenshot(space::ScreenshotExporter::ImageFormat::PNG);
                }

                // Toggle GA on/off with improved state handling
                if (event.key.keysym.scancode == SDL_SCANCODE_G) {
                    if (currentMode == Mode::GENETIC_ALGORITHM && gaRunning) {
                        // Stop GA
                        ga.stop();
                        gaRunning = false;
                        std::cout << "Genetic algorithm stopped." << std::endl;
                    }
                    else {
                        // Start or resume GA - run in main thread!
                        currentMode = Mode::GENETIC_ALGORITHM;
                        gaRunning = true;
                        std::cout << "Genetic algorithm started." << std::endl;
                        ga.run();  // This will block until complete - no thread
                        gaRunning = false;

                        // After GA completes, apply the best solution
                        auto best = ga.getBestSolution();
                        scene.setFrequency(best.frequency);
                        scene.setAmplitude(best.amplitude);
                        scene.setOctaves(best.octaves);

                        // Render the best solution to the default framebuffer
                        glBindFramebuffer(GL_FRAMEBUFFER, 0);  // Bind default framebuffer
                        scene.render();  // This will render to whatever framebuffer is bound
                        window.swap_buffers();

                        std::cout << "Applied best solution: f=" << best.frequency
                            << ", a=" << best.amplitude << ", o=" << best.octaves << std::endl;
                    }
                }

                // Apply best solution
                if (event.key.keysym.scancode == SDL_SCANCODE_B) {
                    auto best = ga.getBestSolution();
                    scene.setFrequency(best.frequency);
                    scene.setAmplitude(best.amplitude);
                    scene.setOctaves(best.octaves);
                    std::cout << "Applied best solution: f=" << best.frequency
                        << ", a=" << best.amplitude << ", o=" << best.octaves << std::endl;
                }

                // Manual parameter adjustments (only when GA not running)
                if (currentMode == Mode::MANUAL || !gaRunning) {
                    // Get current parameters
                    float freq = scene.getFrequency();
                    float amp = scene.getAmplitude();
                    int oct = scene.getOctaves();

                    // Check for keyboard modifiers
                    const Uint8* keyboardState = SDL_GetKeyboardState(nullptr);
                    bool shiftDown = keyboardState[SDL_SCANCODE_LSHIFT] || keyboardState[SDL_SCANCODE_RSHIFT];
                    bool ctrlDown = keyboardState[SDL_SCANCODE_LCTRL] || keyboardState[SDL_SCANCODE_RCTRL];

                    // Frequency adjustment (arrow keys)
                    if (!shiftDown && !ctrlDown) {
                        if (event.key.keysym.scancode == SDL_SCANCODE_UP) {
                            scene.setFrequency(freq + 0.1f);
                            std::cout << "Frequency: " << scene.getFrequency() << std::endl;
                        }
                        else if (event.key.keysym.scancode == SDL_SCANCODE_DOWN) {
                            scene.setFrequency(freq - 0.1f);
                            std::cout << "Frequency: " << scene.getFrequency() << std::endl;
                        }
                    }

                    // Amplitude adjustment (shift + arrow keys)
                    if (shiftDown) {
                        if (event.key.keysym.scancode == SDL_SCANCODE_UP) {
                            scene.setAmplitude(amp + 0.05f);
                            std::cout << "Amplitude: " << scene.getAmplitude() << std::endl;
                        }
                        else if (event.key.keysym.scancode == SDL_SCANCODE_DOWN) {
                            scene.setAmplitude(amp - 0.05f);
                            std::cout << "Amplitude: " << scene.getAmplitude() << std::endl;
                        }
                    }

                    // Octaves adjustment (ctrl + arrow keys)
                    if (ctrlDown) {
                        if (event.key.keysym.scancode == SDL_SCANCODE_UP) {
                            scene.setOctaves(oct + 1);
                            std::cout << "Octaves: " << scene.getOctaves() << std::endl;
                        }
                        else if (event.key.keysym.scancode == SDL_SCANCODE_DOWN) {
                            scene.setOctaves(oct - 1);
                            std::cout << "Octaves: " << scene.getOctaves() << std::endl;
                        }
                    }

                    // Switch to manual mode if any parameter was changed
                    if (freq != scene.getFrequency() ||
                        amp != scene.getAmplitude() ||
                        oct != scene.getOctaves()) {
                        currentMode = Mode::MANUAL;
                    }
                }
            }
        }

        LAST = NOW;
        NOW = SDL_GetPerformanceCounter();
        deltaTime = (double)((NOW - LAST) / (double)SDL_GetPerformanceFrequency());

        scene.update(deltaTime);

        SDL_Delay(5);

        scene.render();
        window.swap_buffers();
    }

    // Clean shutdown
    ga.stop();

    return 0;
}