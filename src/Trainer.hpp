#pragma once

#include "common/Common.hpp"
#include "neuralnetwork/Network.hpp"
#include <vector>


class Trainer {
public:
  virtual ~Trainer() {}

  virtual void Train(
      Network &network, vector<TrainingSample> &trainingSamples, unsigned iterations) = 0;

};
