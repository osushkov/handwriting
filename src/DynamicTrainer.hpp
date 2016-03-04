#pragma once

#include "Trainer.hpp"
#include "neuralnetwork/TrainingProvider.hpp"
#include <random>

class DynamicTrainer : public Trainer {
public:
  DynamicTrainer(float startLearnRate, float epsilonRate, float maxLearnRate, float momentumAmount,
                 unsigned startNumSamples, unsigned maxNumSamples);

  virtual ~DynamicTrainer() = default;

  void Train(Network &network, vector<TrainingSample> &trainingSamples,
             unsigned iterations) override;

  void AddProgressCallback(NetworkTrainerCallback callback) override;

private:
  const float startLearnRate;
  const float epsilonRate;
  const float maxLearnRate;
  const float momentumAmount;
  const unsigned startNumSamples;
  const unsigned maxNumSamples;

  mt19937 rnd;

  unsigned numCompletePasses;
  unsigned curSamplesIndex;
  unsigned curSamplesOffset;
  float curLearnRate;
  float prevSampleError;

  vector<NetworkTrainerCallback> trainingCallbacks;

  void updateLearnRate(unsigned curIter, unsigned iterations, float sampleError);
  TrainingProvider getStochasticSamples(vector<TrainingSample> &allSamples, unsigned curIter,
                                        unsigned totalIters);

  void initWeightGradientRates(Tensor &rates);
  void updateWeightsGradientRates(const Tensor &curGradient, const Tensor &prevGradient,
                                  Tensor &rates);

  unsigned numStochasticSamples(unsigned curIter, unsigned totalIter);
};
