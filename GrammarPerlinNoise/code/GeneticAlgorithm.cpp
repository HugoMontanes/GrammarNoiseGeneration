
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
        minFrequency(0.5f),
        maxFrequency(10.0f),
        minAmplitude(0.1f),
        maxAmplitude(1.0f),
        minOctaves(1),
        maxOctaves(1)
    {
        std::random_device rd;
        rng = std::mt19937(rd());
        uniformDist = std::uniform_real_distribution<float>(0.0f, 1.0f);

        if (scenePtr)
        {
            scenePtr->configureScreenshotPath(screenshotPath);
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
                /*scenePtr->takeScreenshot(ScreenshotExporter::ImageFormat::PNG);*/

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
        //params.octaves = minOctaves + static_cast<int>(uniformDist(rng) * (maxOctaves - minOctaves + 1));


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

    //// Evaluate fitness for all individuals in the population
    //void GeneticAlgorithm::evaluatePopulation() {
    //    // Define number of threads based on hardware
    //    unsigned int numThreads = std::thread::hardware_concurrency();
    //    numThreads = numThreads > 0 ? numThreads : 4; // Default to 4 if detection fails

    //    // Split population into chunks for parallel processing
    //    std::vector<std::thread> threads;
    //    size_t chunkSize = population.size() / numThreads;

    //    for (unsigned int t = 0; t < numThreads; ++t) {
    //        size_t start = t * chunkSize;
    //        size_t end = (t == numThreads - 1) ? population.size() : (t + 1) * chunkSize;

    //        threads.push_back(std::thread([this, start, end]() {
    //            for (size_t i = start; i < end; ++i) {
    //                evaluateIndividual(population[i]);
    //            }
    //            }));
    //    }

    //    // Join all threads
    //    for (auto& thread : threads) {
    //        thread.join();
    //    }

    //    // Find best solution
    //    for (const auto& individual : population) {
    //        if (individual.fitness > bestSolution.fitness) {
    //            bestSolution = individual;
    //        }
    //    }
    //}

    // Sequential version of evaluatePopulation
    void GeneticAlgorithm::evaluatePopulation() {
        // Process each individual sequentially
        for (auto& individual : population) {
            evaluateIndividual(individual);

            // Update best solution if this individual is better
            if (individual.fitness > bestSolution.fitness) {
                bestSolution = individual;
            }
        }
    }

    // New helper method
    void GeneticAlgorithm::evaluateIndividual(VoronoiParameters& individual) 
    {

        //std::lock_guard<std::mutex> lock(sceneMutex); // Add mutex as class member

        if (scenePtr) {
            scenePtr->setFrequency(individual.frequency);
            scenePtr->setAmplitude(individual.amplitude);
            scenePtr->setOctaves(individual.octaves);

            scenePtr->renderToFBO();

            //Take screenshot
            bool screenshotTaken = scenePtr->takeScreenshot(ScreenshotExporter::ImageFormat::PNG);

            if (screenshotTaken) {
                
                // Get the last screenshot number and construct the path
                int lastImageCounter = scenePtr->getScreenshotExporter()->getLastImageCounter();
                screenshotPath = scenePtr->getScreenshotExporter()->getOutputPath() + "image_" + std::to_string(lastImageCounter) + ".png";

                // Verify file exists before evaluating
                std::ifstream file(screenshotPath);
                if (file.good()) {
                    individual.fitness = fitnessEvaluator(individual, screenshotPath);
                    file.close();
                }
                else {
                    std::cerr << "Screenshot file not found: " << screenshotPath << std::endl;
                    individual.fitness = 0.0f;
                }
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

        //// Mutate octaves (discrete parameter)
        //if (uniformDist(rng) < currentMutationRate) {
        //    // Use a more nuanced approach for discrete parameter
        //    if (uniformDist(rng) < 0.5f) {
        //        // 50% chance to increment or decrement by 1
        //        int change = (uniformDist(rng) < 0.5f) ? -1 : 1;
        //        individual.octaves = std::clamp(individual.octaves + change, minOctaves, maxOctaves);
        //    }
        //    else {
        //        // 50% chance to set to random value within range
        //        individual.octaves = minOctaves + static_cast<int>(uniformDist(rng) * (maxOctaves - minOctaves + 1));
        //    }
        //}
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
        maintainPopulationDiversity();
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
    
    void GeneticAlgorithm::resetWithSeed(const VoronoiParameters& seedParams) 
    {
        population.clear();
        population.reserve(populationSize);

        // Add the seed parameters
        population.push_back(seedParams);

        // Create variations around the seed
        for (size_t i = 1; i < populationSize; ++i) {
            VoronoiParameters variation = seedParams;
            // Create small variations
            variation.frequency += (uniformDist(rng) * 2.0f - 1.0f) * (maxFrequency - minFrequency) * 0.1f;
            variation.amplitude += (uniformDist(rng) * 2.0f - 1.0f) * (maxAmplitude - minAmplitude) * 0.1f;

            // Occasionally change octaves
            /*if (uniformDist(rng) < 0.3f) {
                int change = (uniformDist(rng) < 0.5f) ? -1 : 1;
                variation.octaves = std::clamp(variation.octaves + change, minOctaves, maxOctaves);
            }*/

            // Ensure constraints
            variation.frequency = std::clamp(variation.frequency, minFrequency, maxFrequency);
            variation.amplitude = std::clamp(variation.amplitude, minAmplitude, maxAmplitude);

            population.push_back(variation);
        }

        currentGeneration = 0;
        bestSolution = seedParams;
        bestSolution.fitness = -1.0f; // Reset fitness
    }

    // Add diversity preservation method to GeneticAlgorithm class
    void GeneticAlgorithm::maintainPopulationDiversity() {
        // Calculate population statistics
        float avgFitness = 0.0f;
        for (const auto& individual : population) {
            avgFitness += individual.fitness;
        }
        avgFitness /= population.size();

        // Inject new random individuals if diversity is low
        float diversityThreshold = 0.1f; // Adjust based on your fitness scale
        bool needsDiversity = false;

        // Check if fitness values are too similar
        float fitnessVariance = 0.0f;
        for (const auto& individual : population) {
            fitnessVariance += (individual.fitness - avgFitness) * (individual.fitness - avgFitness);
        }
        fitnessVariance /= population.size();

        if (fitnessVariance < diversityThreshold) {
            // Replace 10% of population with new random individuals
            int replacementCount = std::max(1, int(population.size() * 0.1));
            for (int i = 0; i < replacementCount; i++) {
                int idx = int(uniformDist(rng) * population.size());
                population[idx] = generateRandomIndividual();
            }
        }
    }
    void GeneticAlgorithm::setScreenshotPath(const std::string& path)
    {
        screenshotPath = path;
        if (scenePtr)
        {
            scenePtr->configureScreenshotPath(path);
        }
    }
}