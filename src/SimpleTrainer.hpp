#pragma once

#include "Trainer.hpp"
#include "neuralnetwork/TrainingProvider.hpp"


class SimpleTrainer : public Trainer {
public:

  SimpleTrainer(float startLearnRate, float endLearnRate, unsigned stochasticSamples);
  virtual ~SimpleTrainer() = default;

  void Train(
      Network &network, vector<TrainingSample> &trainingSamples, unsigned iterations) override;

private:

  const float startLearnRate;
  const float endLearnRate;
  const unsigned stochasticSamples;

  unsigned curSamplesIndex;

  float getLearnRate(unsigned curIter, unsigned iterations);
  TrainingProvider getStochasticSamples(vector<TrainingSample> &allSamples);

};
