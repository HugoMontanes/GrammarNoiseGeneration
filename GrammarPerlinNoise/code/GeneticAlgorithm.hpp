
#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <memory>
#include <string>
#include <functional>
#include "Scene.hpp"

//Forward declaration ONNXModel
class ONNXModel;

namespace space
{
    struct VoronoiParameters
    {
        float frequency;
        float amplitude;
        int octaves;
        float fitness;

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
        VoronoiParameters tournamentSelection(size_t tournamentSize);
        std::pair<VoronoiParameters, VoronoiParameters> crossover(const VoronoiParameters& parent1, const VoronoiParameters& parent2);
        void mutate(VoronoiParameters& individual);

    public:


    };
}