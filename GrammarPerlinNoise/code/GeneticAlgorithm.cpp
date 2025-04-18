
#include "GeneticAlgorithm.hpp"
#include "ONNXModel.hpp"
#include <algorithm>
#include <iostream>
#include <ctime>
#include <cmath>
#include <numeric>
#include <mutex>

namespace space
{
    GeneticAlgorithm::GeneticAlgorithm
    (
        Scene* scene,
        size_t popSize,
        float crossover,
        float mutation,
        int maxGen,
        const std::string& onnxModelPath
    ) :
        scenePtr(scene),
        populationSize(popSize),
        crossoverRate(crossover),
        mutationRate(mutation),
        maxGenerations(maxGen),
        currentGeneration(0),
        isRunning(false),
        minFrequency(1.0f),
        maxFrequency(10.0f),
        minAmplitude(0.1f),
        maxAmplitude(1.0f),
        minOctaves(1),
        maxOctaves(5)
    {
        std::random_device rd;
        rng = std::mt19937(rd());
        uniformDist = std::uniform_real_distribution<float>(0.0f, 1.0f);

        if (scenePtr)
        {
            scenePtr->configureScreenshotPath("../../../assets/generated_images");
        }

        // Load the ONNX model for fitness evaluation
        onnxModel = std::make_unique<ONNXModel>(onnxModelPath);

        fitnessEvaluator = [this](const VoronoiParameters& params, const std::string& screenshotPath)
        {
                // Update shader parameters
            if (scenePtr)
            {
                scenePtr->setFrequency(params.frequency);
                scenePtr->setAmplitude(params.amplitude);
                scenePtr->setOctaves(params.octaves);

                // Render scene with these parameters
                scenePtr->render();

                // Take screenshot to evaluate
                scenePtr->takeScreenshot();

                // Use ONNX model to evaluate the screenshot
                float fitness = onnxModel->evaluateImage(screenshotPath);
                return fitness;
            }
            return 0.0f;
        };
    }

    VoronoiParameters GeneticAlgorithm::generateRandomIndividual()
    {
        VoronoiParameters params;

        //Generate random values within specified ranges
        params.frequency = minFrequency + uniformDist(rng) * (maxFrequency - minFrequency);
        params.amplitude = minAmplitude + uniformDist(rng) * (maxAmplitude - minAmplitude);
        params.octaves = minOctaves + static_cast<int>(uniformDist(rng) * (maxOctaves - minOctaves + 1));


        return params;
    }

    void GeneticAlgorithm::initializePopulation()
    {
        population.clear();
        population.reserve(populationSize);

        for (size_t i = 0; i < populationSize; ++i)
        {
            population.push_back(generateRandomIndividual());
        }

        currentGeneration = 0;
        bestSolution = VoronoiParameters(); //Reset best solution
        bestSolution.fitness = -1.0f; // Initialize with worst possible fitness
    }

    // Evaluate fitness for all individuals in the population
    void GeneticAlgorithm::evaluatePopulation() {
        // First, render and capture screenshots for all individuals on the main thread
        for (auto& individual : population) {
            if (scenePtr) {
                scenePtr->setFrequency(individual.frequency);
                scenePtr->setAmplitude(individual.amplitude);
                scenePtr->setOctaves(individual.octaves);

                scenePtr->render();
                scenePtr->takeScreenshot();

                // Store the screenshot path for this individual
                int lastImageCounter = scenePtr->getScreenshotExporter()->getLastImageCounter();
                individual.screenshotPath = "../../../assets/generated_images/image_" +
                    std::to_string(lastImageCounter) + ".png";
            }
        }

        // Now evaluate fitness in parallel (without rendering)
        unsigned int numThreads = std::thread::hardware_concurrency();
        numThreads = numThreads > 0 ? numThreads : 4;

        std::vector<std::thread> threads;
        size_t chunkSize = population.size() / numThreads;

        for (unsigned int t = 0; t < numThreads; ++t) {
            size_t start = t * chunkSize;
            size_t end = (t == numThreads - 1) ? population.size() : (t + 1) * chunkSize;

            threads.push_back(std::thread([this, start, end]() {
                for (size_t i = start; i < end; ++i) {
                    // Only evaluate fitness, no rendering or screenshot taking
                    population[i].fitness = fitnessEvaluator(population[i], population[i].screenshotPath);
                }
                }));
        }

        // Join threads
        for (auto& thread : threads) {
            thread.join();
        }

        // Find best solution
        for (const auto& individual : population) {
            if (individual.fitness > bestSolution.fitness) {
                bestSolution = individual;
            }
        }
    }

    // New helper method
    void GeneticAlgorithm::evaluateIndividual(VoronoiParameters& individual) 
    {

        std::lock_guard<std::mutex> lock(sceneMutex); // Add mutex as class member

        if (scenePtr) {
            scenePtr->setFrequency(individual.frequency);
            scenePtr->setAmplitude(individual.amplitude);
            scenePtr->setOctaves(individual.octaves);

            scenePtr->render();

            std::string screenshotPath;
            bool screenshotTaken = scenePtr->takeScreenshot();

            if (screenshotTaken) {
                
                // Get the last screenshot number and construct the path
                int lastImageCounter = scenePtr->getScreenshotExporter()->getLastImageCounter();
                screenshotPath = "../../../assets/generated_images/image_" + std::to_string(lastImageCounter) + ".png";
                individual.fitness = fitnessEvaluator(individual, screenshotPath);
            }
            else
            {
                std::cerr << "Failed to take screenshot for individual" << std::endl;
                individual.fitness = 0.0f;
            }
        }
    }

    VoronoiParameters GeneticAlgorithm::tournamentSelection(size_t tournamentSize)
    {
        std::vector<VoronoiParameters> tournament;
        tournament.reserve(tournamentSize);

        // Select random individuals for the tournament
        for (size_t i = 0; i < tournamentSize; ++i)
        {
            size_t randomIndex = static_cast<size_t>(uniformDist(rng) * population.size());
            tournament.push_back(population[randomIndex]);
        }

        // Find the winner (individual with highest fitness)
        VoronoiParameters winner = tournament[0];
        for (size_t i = 1; i < tournament.size(); ++i)
        {
            if (tournament[i].fitness > winner.fitness)
            {
                winner = tournament[i];
            }
        }

        return winner;

    }
    std::pair<VoronoiParameters, VoronoiParameters> GeneticAlgorithm::crossover(const VoronoiParameters& parent1, const VoronoiParameters& parent2)
    {
        if (uniformDist(rng) > crossoverRate)
        {
            return{ parent1, parent2 };
        }

        //Perform weighted average (blend) crossover
        float alpha = uniformDist(rng);

        VoronoiParameters child1, child2;

        // Blend frequency
        child1.frequency = alpha * parent1.frequency + (1.0f - alpha) * parent2.frequency;
        child2.frequency = alpha * parent2.frequency + (1.0f - alpha) * parent1.frequency;

        // Blend amplitude
        child1.amplitude = alpha * parent1.amplitude + (1.0f - alpha) * parent2.amplitude;
        child2.amplitude = alpha * parent2.amplitude + (1.0f - alpha) * parent1.amplitude;

        // For discrete parameter (octaves), use probability-based selection
        child1.octaves = (uniformDist(rng) < alpha) ? parent1.octaves : parent2.octaves;
        child2.octaves = (uniformDist(rng) < alpha) ? parent2.octaves : parent1.octaves;

        return { child1, child2 };
    }

    void GeneticAlgorithm::mutate(VoronoiParameters& individual, float adaptiveMutationRate) 
    {
        // Use the passed adaptive rate or fall back to default
        float currentMutationRate = (adaptiveMutationRate > 0) ? adaptiveMutationRate : mutationRate;

        // Mutate frequency
        if (uniformDist(rng) < currentMutationRate) {
            // Adaptive mutation strength - larger early, smaller later
            float mutationStrength = 0.3f * (1.0f - float(currentGeneration) / maxGenerations) + 0.1f;
            float mutation = (uniformDist(rng) * 2.0f - 1.0f) * mutationStrength * (maxFrequency - minFrequency);
            individual.frequency = std::clamp(individual.frequency + mutation, minFrequency, maxFrequency);
        }

        // Mutate amplitude
        if (uniformDist(rng) < currentMutationRate) {
            float mutationStrength = 0.3f * (1.0f - float(currentGeneration) / maxGenerations) + 0.1f;
            float mutation = (uniformDist(rng) * 2.0f - 1.0f) * mutationStrength * (maxAmplitude - minAmplitude);
            individual.amplitude = std::clamp(individual.amplitude + mutation, minAmplitude, maxAmplitude);
        }

        // Mutate octaves (discrete parameter)
        if (uniformDist(rng) < currentMutationRate) {
            // Use a more nuanced approach for discrete parameter
            if (uniformDist(rng) < 0.5f) {
                // 50% chance to increment or decrement by 1
                int change = (uniformDist(rng) < 0.5f) ? -1 : 1;
                individual.octaves = std::clamp(individual.octaves + change, minOctaves, maxOctaves);
            }
            else {
                // 50% chance to set to random value within range
                individual.octaves = minOctaves + static_cast<int>(uniformDist(rng) * (maxOctaves - minOctaves + 1));
            }
        }
    }

    // Evolve the population by one generation
    void GeneticAlgorithm::evolvePopulation() 
    {
        std::vector<VoronoiParameters> newPopulation;
        newPopulation.reserve(populationSize);

        // Elitism: Keep the best individuals (top 10%)
        size_t elitismCount = std::max(size_t(1), populationSize / 10);

        // Sort population by fitness in descending order
        std::vector<VoronoiParameters> sortedPopulation = population;
        std::sort(sortedPopulation.begin(), sortedPopulation.end(),
            [](const VoronoiParameters& a, const VoronoiParameters& b) {
                return a.fitness > b.fitness;
            });

        // Add elite individuals to new population
        for (size_t i = 0; i < elitismCount && i < sortedPopulation.size(); ++i) {
            newPopulation.push_back(sortedPopulation[i]);
        }

        // Generate rest of new population
        while (newPopulation.size() < populationSize) {
            // Select parents
            VoronoiParameters parent1 = tournamentSelection(3); // Tournament size of 3
            VoronoiParameters parent2 = tournamentSelection(3);

            // Avoid identical parents to increase diversity
            int attempts = 0;
            while (parent1.frequency == parent2.frequency &&
                parent1.amplitude == parent2.amplitude &&
                parent1.octaves == parent2.octaves &&
                attempts < 5) {
                parent2 = tournamentSelection(3);
                attempts++;
            }

            // Crossover
            auto [child1, child2] = crossover(parent1, parent2);

            // Mutate children - with adaptive mutation rate
            // Increase mutation slightly in later generations to escape local optima
            float adaptiveMutationRate = mutationRate * (1.0f + float(currentGeneration) / maxGenerations);
            mutate(child1, adaptiveMutationRate);
            mutate(child2, adaptiveMutationRate);

            // Add to new population
            newPopulation.push_back(child1);
            if (newPopulation.size() < populationSize) {
                newPopulation.push_back(child2);
            }
        }

        // Replace old population with new one
        population = std::move(newPopulation);
        currentGeneration++;
    }

    // Run the genetic algorithm for a specified number of generations
    void GeneticAlgorithm::run() 
    {
        isRunning = true;

        std::cout << "Starting Genetic Algorithm optimization..." << std::endl;

        // Initialize with random population
        initializePopulation();

        // Main GA loop
        while (currentGeneration < maxGenerations && isRunning) {
            std::cout << "Generation " << currentGeneration << "/" << maxGenerations << std::endl;

            // Evaluate current population
            evaluatePopulation();

            // Print best solution of this generation
            std::cout << "Best fitness: " << bestSolution.fitness
                << " (f=" << bestSolution.frequency
                << ", a=" << bestSolution.amplitude
                << ", o=" << bestSolution.octaves << ")" << std::endl;

            // Evolve to next generation
            evolvePopulation();
        }

        // Set scene to use best parameters found
        if (scenePtr) {
            scenePtr->setFrequency(bestSolution.frequency);
            scenePtr->setAmplitude(bestSolution.amplitude);
            scenePtr->setOctaves(bestSolution.octaves);

            std::cout << "Optimization complete!" << std::endl;
            std::cout << "Best solution:" << std::endl;
            std::cout << "  Frequency: " << bestSolution.frequency << std::endl;
            std::cout << "  Amplitude: " << bestSolution.amplitude << std::endl;
            std::cout << "  Octaves: " << bestSolution.octaves << std::endl;
            std::cout << "  Fitness: " << bestSolution.fitness << std::endl;
        }

        isRunning = false;
    }

    // Stop the algorithm if it's running
    void GeneticAlgorithm::stop() 
    {
        isRunning = false;
    }

    // Get current best solution
    VoronoiParameters GeneticAlgorithm::getBestSolution() const 
    {
        return bestSolution;
    }

    // Get current generation number
    int GeneticAlgorithm::getCurrentGeneration() const 
    {
        return currentGeneration;
    }

    // Set parameter constraints
    void GeneticAlgorithm::setParameterConstraints(
        float minFreq, float maxFreq,
        float minAmp, float maxAmp,
        int minOct, int maxOct) {

        minFrequency = minFreq;
        maxFrequency = maxFreq;
        minAmplitude = minAmp;
        maxAmplitude = maxAmp;
        minOctaves = minOct;
        maxOctaves = maxOct;
    }
}