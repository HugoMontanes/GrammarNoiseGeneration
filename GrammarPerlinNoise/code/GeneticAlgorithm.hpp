/*
* Codigo realizado por Hugo Montañés García.
*/

#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <memory>
#include <string>
#include <functional>
#include <mutex>
#include "Scene.hpp"
#include "ONNXModel.hpp"

namespace space
{
    struct VoronoiParameters
    {
        float frequency;
        float amplitude;
        int octaves;
        float fitness;
        /*std::string screenshotPath;*/

        VoronoiParameters(float freq = 2.5f, float amp = 0.4f, int oct = 1)
            : frequency(freq), amplitude(amp), octaves(oct), fitness(0.0f){}
    };

    class GeneticAlgorithm
    {
    private:

        //GA Parameters
        size_t populationSize;
        float crossoverRate;
        float mutationRate;
        int maxGenerations;
        int currentGeneration;

        //Parameter constraints
        float minFrequency, maxFrequency;
        float minAmplitude, maxAmplitude;
        int minOctaves, maxOctaves;

        std::string screenshotPath;

        //Random generators
        std::mt19937 rng;
        std::uniform_real_distribution<float> uniformDist;

        std::vector<VoronoiParameters> population;

        Scene* scenePtr;

        std::unique_ptr<ONNXModel> onnxModel;

        bool isRunning;

        VoronoiParameters bestSolution;


        using FitnessFunction = std::function<float(const VoronoiParameters&, const std::string& screenshotPath)>;
        FitnessFunction fitnessEvaluator;

        VoronoiParameters generateRandomIndividual();
        void evaluateIndividual(VoronoiParameters& individual);
        VoronoiParameters tournamentSelection(size_t tournamentSize);
        std::pair<VoronoiParameters, VoronoiParameters> crossover(const VoronoiParameters& parent1, const VoronoiParameters& parent2);
        void mutate(VoronoiParameters& individual, float adaptiveMutationRate = 0.0f);

        std::mutex sceneMutex;

    public:

        GeneticAlgorithm
        (
            Scene* scene, 
            size_t popSize = 50, 
            float crossover = 0.8f, 
            float mutation = 0.2f, 
            int maxGen = 20, const std::string& onnxModelPath = "../../../assets/models/texture_discriminator.onnx"
        );

        //Core methods
        void initializePopulation();
        void evaluatePopulation();
        void evolvePopulation();
        void run();
        void stop();

        //Getters and Setters
        VoronoiParameters getBestSolution() const;
        int getCurrentGeneration() const;
		void setParameterConstraints(
			float minFreq, float maxFreq,
			float minAmp, float maxAmp,
			int minOct, int maxOct
		);

        void resetWithSeed(const VoronoiParameters& seedParams);
        void maintainPopulationDiversity();

        void setScreenshotPath(const std::string& path);
    };
}