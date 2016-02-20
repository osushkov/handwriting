#pragma once

#include "TrainingProvider.hpp"
#include "../common/Common.hpp"
#include "../common/Math.hpp"
#include "Tensor.hpp"
#include <vector>


class Network {
public:
  static void OutputDebugging(void);

  Network(const vector<unsigned> &layerSizes);
  virtual ~Network();

  Vector Process(const Vector &input);
  pair<Tensor, float> ComputeGradient(const TrainingProvider &samplesProvider);
  void ApplyUpdate(const Tensor &weightUpdates);

  std::ostream& Output(std::ostream& stream);

private:
  struct NetworkImpl;
  uptr<NetworkImpl> impl;
};
