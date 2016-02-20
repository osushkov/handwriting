
#include "Network.hpp"
#include "../util/Util.hpp"
#include "../common/ThreadPool.hpp"
#include <cassert>
#include <cmath>
#include <future>
#include <iostream>
#include <thread>
#include <atomic>


static const float INIT_WEIGHT_RANGE = 0.1f;


struct NetworkContext {
  vector<Vector> layerOutputs;
  vector<Vector> layerDeltas;
};


struct Network::NetworkImpl {
  unsigned numInputs;
  unsigned numOutputs;
  unsigned numLayers;

  Tensor layerWeights;
  Tensor zeroGradient;


  NetworkImpl(const vector<unsigned> &layerSizes) {
    assert(layerSizes.size() >= 2);
    this->numLayers = layerSizes.size() - 1;
    this->numInputs = layerSizes[0];
    this->numOutputs = layerSizes[layerSizes.size() - 1];

    for (unsigned i = 0; i < numLayers; i++) {
      layerWeights.AddLayer(createLayer(layerSizes[i], layerSizes[i+1]));
    }

    zeroGradient = layerWeights;
    for (unsigned i = 0; i < zeroGradient.NumLayers(); i++) {
      zeroGradient(i).setZero();
    }
  }

  Vector Process(const Vector &input) {
    assert(input.rows() == numInputs);

    NetworkContext ctx;
    return process(input, ctx);
  }

  pair<Tensor, float> ComputeGradient(const TrainingProvider &samplesProvider) {
    // Single threaded code below

    // auto gradient = make_pair(zeroGradient, 0.0f);
    // Tensor& netGradient{gradient.first};
    // float& error{gradient.second};
    //
    // for (unsigned i = 0; i < NUM_THREADS; i++) {
    //   unsigned start = (i * samplesProvider.NumSamples()) / NUM_THREADS;
    //   unsigned end = ((i+1) * samplesProvider.NumSamples()) / NUM_THREADS;
    //
    //   auto subsetGradient = computeGradientSubset(samplesProvider, start, end);
    //   netGradient += subsetGradient.first;
    //   error += subsetGradient.second;
    // }

    const unsigned numSubsets = ThreadPool::instance().NumThreads();

    auto gradient = make_pair(zeroGradient, 0.0f);
    Tensor& netGradient = gradient.first;
    float& error = gradient.second;

    mutex gradientMutex;
    vector<future<void>> futures;
    futures.reserve(numSubsets);

    for (unsigned i = 0; i < numSubsets; i++) {
      futures.push_back(ThreadPool::instance().Execute(
          [this, &gradientMutex, &samplesProvider, &netGradient, &error, i, numSubsets]() {
        unsigned start = (i * samplesProvider.NumSamples()) / numSubsets;
        unsigned end = ((i+1) * samplesProvider.NumSamples()) / numSubsets;
        auto subsetGradient = computeGradientSubset(samplesProvider, start, end);

        {
          std::unique_lock<std::mutex> lock(gradientMutex);
          netGradient += subsetGradient.first;
          error += subsetGradient.second;
        }
      }));
    }

    for (auto& f : futures) {
      f.get();
    }

    float scaleFactor = 1.0f / samplesProvider.NumSamples();
    netGradient *= scaleFactor;
    error *= scaleFactor;

    return gradient;
  }

  void ApplyUpdate(const Tensor &weightUpdates) {
    layerWeights += weightUpdates;
  }

private:

  Matrix createLayer(unsigned inputSize, unsigned layerSize) {
    assert(inputSize > 0 && layerSize > 0);

    unsigned numRows = layerSize;
    unsigned numCols = inputSize + 1; // +1 accounts for bias input

    Matrix result(numRows, numCols);

    for (unsigned r = 0; r < result.rows(); r++) {
      for (unsigned c = 0; c < result.cols(); c++) {
        result(r, c) = Util::RandInterval(-INIT_WEIGHT_RANGE, INIT_WEIGHT_RANGE);
      }
    }

    return result;
  }

  pair<Tensor, float> computeGradientSubset(
      const TrainingProvider &samplesProvider, unsigned start, unsigned end) {

    auto gradient = make_pair(zeroGradient, 0.0f);
    Tensor& netGradient = gradient.first;
    float& error = gradient.second;

    NetworkContext ctx;
    for (unsigned i = start; i < end; i++) {
      pair<Tensor, float> gradientAndError =
          computeSampleGradient(samplesProvider.GetSample(i), ctx);
      netGradient += gradientAndError.first;
      error += gradientAndError.second;
    }

    return gradient;
  }

  Vector process(const Vector &input, NetworkContext &ctx) {
    assert(input.rows() == numInputs);

    ctx.layerOutputs.resize(layerWeights.NumLayers());
    ctx.layerOutputs[0] = getLayerOutput(input, layerWeights(0));
    for (unsigned i = 1; i < layerWeights.NumLayers(); i++) {
      ctx.layerOutputs[i] = getLayerOutput(ctx.layerOutputs[i-1], layerWeights(i));
    }

    assert(ctx.layerOutputs[ctx.layerOutputs.size()-1].rows() == numOutputs);
    return ctx.layerOutputs[ctx.layerOutputs.size()-1];
  }

  Vector getLayerOutput(const Vector &prevLayer, const Matrix &layerWeights) {
    Vector z = layerWeights.topRightCorner(layerWeights.rows(), layerWeights.cols()-1) * prevLayer;
    for (unsigned i = 0; i < layerWeights.rows(); i++) {
      z(i) += layerWeights(i, 0);
      z(i) = activationFunc(z(i));
    }
    return z;
  }

  float activationFunc(float v) {
    return 1.0f / (1.0f + expf(-v));
  }

  pair<Tensor, float> computeSampleGradient(const TrainingSample &sample, NetworkContext &ctx) {
    Vector output = process(sample.input, ctx);

    ctx.layerDeltas.resize(numLayers);
    ctx.layerDeltas[ctx.layerDeltas.size() - 1] = output - sample.expectedOutput; // cross entropy error function.

    for (int i = ctx.layerDeltas.size() - 2; i >= 0; i--) {
      Matrix noBiasWeights =
          layerWeights(i+1).bottomRightCorner(layerWeights(i+1).rows(), layerWeights(i+1).cols()-1);
      ctx.layerDeltas[i] = noBiasWeights.transpose() * ctx.layerDeltas[i+1];

      assert(ctx.layerDeltas[i].rows() == ctx.layerOutputs[i].rows());
      for (unsigned r = 0; r < ctx.layerDeltas[i].rows(); r++) {
        float out = ctx.layerOutputs[i](r);
        ctx.layerDeltas[i](r) *= out * (1.0f - out);
      }
    }

    auto result = make_pair(Tensor(), 0.0f);
    for (unsigned i = 0; i < numLayers; i++) {
      auto inputs = getInputWithBias(i == 0 ? sample.input : ctx.layerOutputs[i-1]);
      result.first.AddLayer(ctx.layerDeltas[i] * inputs.transpose());
    }

    for (unsigned i = 0; i < output.rows(); i++) {
      result.second += (output(i) - sample.expectedOutput(i)) * (output(i) - sample.expectedOutput(i));
    }

    return result;
  }

  Vector getInputWithBias(const Vector &noBiasInput) {
    Vector result(noBiasInput.rows() + 1);
    result(0) = 1.0f;
    result.bottomRightCorner(noBiasInput.rows(), 1) = noBiasInput;
    return result;
  }
};


Network::Network(const vector<unsigned> &layerSizes) : impl(new NetworkImpl(layerSizes)) {}
Network::~Network() = default;

Vector Network::Process(const Vector &input) {
  return impl->Process(input);
}

pair<Tensor, float> Network::ComputeGradient(const TrainingProvider &samplesProvider) {
  return impl->ComputeGradient(samplesProvider);
}

void Network::ApplyUpdate(const Tensor &weightUpdates) {
  impl->ApplyUpdate(weightUpdates);
}

std::ostream& Network::Output(std::ostream& stream) {
  for (unsigned i = 0; i < impl->layerWeights.NumLayers(); i++) {
    for (unsigned r = 0; r < impl->layerWeights(i).rows(); r++) {
      for (unsigned c = 0; c < impl->layerWeights(i).cols(); c++) {
        stream << impl->layerWeights(i)(r, c) << " ";
      }
      stream << endl;
    }
    stream << endl;
  }
  return stream;
}
