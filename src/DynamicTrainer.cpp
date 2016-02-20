
#include "DynamicTrainer.hpp"
#include <cassert>
#include <iostream>
#include <random>


DynamicTrainer::DynamicTrainer(float startLearnRate,
                               float maxLearnRate,
                               float momentumAmount,
                               unsigned stochasticSamples) :
    startLearnRate(startLearnRate),
    maxLearnRate(maxLearnRate),
    momentumAmount(momentumAmount),
    stochasticSamples(stochasticSamples) {

  assert(startLearnRate > 0.0f);
  assert(maxLearnRate > 0.0f);
  assert(momentumAmount >= 0.0f && momentumAmount < 1.0f);
  assert(stochasticSamples > 0);

  random_device rd;
  this->rnd = mt19937(rd());
}

void DynamicTrainer::Train(
    Network &network, vector<TrainingSample> &trainingSamples, unsigned iterations) {

  shuffle(trainingSamples.begin(), trainingSamples.end(), rnd);
  numCompletePasses = 0;
  curSamplesIndex = 0;
  curSamplesOffset = 0;

  curLearnRate = startLearnRate;
  prevSampleError = 0.0f;

  Tensor momentum;
  for (unsigned i = 0; i < iterations; i++) {
    // if (i%1000 == 0) {
    //   cout << i << "/" << iterations << endl;
    // }

    TrainingProvider samplesProvider = getStochasticSamples(trainingSamples);
    pair<Tensor, float> gradientError = network.ComputeGradient(samplesProvider);

    gradientError.first *= -curLearnRate;

    if (i == 0) {
      momentum = gradientError.first;
    } else {
      momentum = momentum*momentumAmount + gradientError.first*(1.0f - momentumAmount);
    }

    network.ApplyUpdate(momentum);
    updateLearnRate(i, iterations, gradientError.second);
  }
}

void DynamicTrainer::updateLearnRate(unsigned curIter, unsigned iterations, float sampleError) {
  if (curIter > 0) {
    if (sampleError < prevSampleError) {
      curLearnRate *= 1.1f;
      curLearnRate = min<float>(curLearnRate, maxLearnRate);
    } else {
      curLearnRate *= 0.95f;
    }
  }

  prevSampleError = sampleError;
}

TrainingProvider DynamicTrainer::getStochasticSamples(vector<TrainingSample> &allSamples) {
  unsigned numSamples = min<unsigned>(allSamples.size(), stochasticSamples);

  if ((curSamplesIndex + numSamples) > allSamples.size()) {
    if (numCompletePasses%10 == 0) {
      shuffle(allSamples.begin(), allSamples.end(), rnd);
    } else {
      curSamplesOffset = rnd() % allSamples.size();
    }
    curSamplesIndex = 0;
    numCompletePasses++;
  }

  auto result = TrainingProvider(allSamples, numSamples, curSamplesIndex + curSamplesOffset);
  curSamplesIndex += numSamples;

  return result;
}
