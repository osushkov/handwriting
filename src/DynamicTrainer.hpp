#pragma once

#include "Trainer.hpp"
#include "neuralnetwork/TrainingProvider.hpp"
#include <random>

class DynamicTrainer : public Trainer {
public:

  DynamicTrainer(float startLearnRate,
                 float maxLearnRate,
                 float momentumAmount,
                 unsigned stochasticSamples);

  virtual ~DynamicTrainer() = default;

  void Train(
      Network &network, vector<TrainingSample> &trainingSamples, unsigned iterations) override;

private:

  const float startLearnRate;
  const float maxLearnRate;
  const float momentumAmount;
  const unsigned stochasticSamples;

  mt19937 rnd;

  unsigned numCompletePasses;
  unsigned curSamplesIndex;
  unsigned curSamplesOffset;
  float curLearnRate;
  float prevSampleError;

  void updateLearnRate(unsigned curIter, unsigned iterations, float sampleError);
  TrainingProvider getStochasticSamples(vector<TrainingSample> &allSamples);
};
