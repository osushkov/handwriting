#pragma once

#include "TrainingSample.hpp"
#include "../common/Common.hpp"
#include <vector>
#include <cassert>


class TrainingProvider {
public:

  TrainingProvider(
      const vector<TrainingSample> &allSamples,
      unsigned numSamples,
      unsigned offset) :
        allSamples(allSamples),
        numSamples(numSamples),
        offset(offset) {}

  const TrainingSample& GetSample(unsigned index) const {
    assert(index < numSamples);
    unsigned i = (index + offset) % allSamples.size();
    return allSamples[i];
  }

  unsigned NumSamples(void) const {
    return numSamples;
  }

private:
  const vector<TrainingSample> &allSamples;
  unsigned numSamples;
  unsigned offset;
};
