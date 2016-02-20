
#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <Eigen/Dense>

// #include "common/ThreadPool.hpp"
#include "util/Util.hpp"
#include "neuralnetwork/Network.hpp"
#include "SimpleTrainer.hpp"
#include "DynamicTrainer.hpp"


using namespace std;
using Eigen::MatrixXd;


vector<TrainingSample> getTrainingData(unsigned howMany) {
  vector<TrainingSample> trainingData;
  trainingData.reserve(howMany);

  float centreX = 0.75f, centreY = 0.6f;
  float radius = 0.4f;
  for (unsigned i = 0; i < howMany; i++) {
    Vector input(2);
    input(0) = Util::RandInterval(-1.0, 1.0);
    input(1) = Util::RandInterval(-1.0, 1.0);

    float d = sqrt((input(0) - centreX) * (input(0) - centreX) +
        (input(1) - centreY) * (input(1) - centreY));

    Vector output(1);
    output(0) = d < radius ? 1.0f : 0.0f;

    trainingData.push_back(TrainingSample{input, output});
  }

  // Vector input(2);
  // Vector output(1);
  //
  // input(0) = 1.0; input(1) = 0.0; output(0) = 1.0;
  // trainingData.push_back(TrainingSample{input, output});
  //
  // input(0) = 0.0; input(1) = 1.0; output(0) = 1.0;
  // trainingData.push_back(TrainingSample{input, output});
  //
  // input(0) = 1.0; input(1) = 1.0; output(0) = 0.0;
  // trainingData.push_back(TrainingSample{input, output});
  //
  // input(0) = 0.0; input(1) = 0.0; output(0) = 0.0;
  // trainingData.push_back(TrainingSample{input, output});

  return trainingData;
}

void evaluateNetwork(Network &network, const std::vector<TrainingSample> &evalSamples) {
  unsigned numCorrect = 0;

  for (const auto& es : evalSamples) {
    auto result = network.Process(es.input);
    // cout << es << " -> " << result << endl << endl;

    for (unsigned i = 0; i < result.rows(); i++) {
      bool isCorrect =
          (result(i) > 0.6f && es.expectedOutput(i) > 0.5f) ||
          (result(i) < 0.4f && es.expectedOutput(i) < 0.5f);
      numCorrect += isCorrect ? 1 : 0;
    }
  }

  cout << "frac correct: " << (numCorrect / (float) evalSamples.size()) << endl;
}

int main() {
  srand(1234);

  Network network({2, 3, 1});
  // uptr<Trainer> trainer = make_unique<SimpleTrainer>(0.2, 0.001, 500);
  uptr<Trainer> trainer = make_unique<DynamicTrainer>(0.5f, 0.5f, 0.25f, 500);

  vector<TrainingSample> trainingSamples = getTrainingData(8000);
  trainer->Train(network, trainingSamples, 100000);

  vector<TrainingSample> evalSamples = getTrainingData(1000);
  evaluateNetwork(network, evalSamples);

  cout << "finished" << endl;
  return 0;
}
