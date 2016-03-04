
#include <Eigen/Core>
#include <Eigen/Dense>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// #include "common/ThreadPool.hpp"
#include "DynamicTrainer.hpp"
#include "SimpleTrainer.hpp"
#include "common/Common.hpp"
#include "image/IdxImages.hpp"
#include "image/IdxLabels.hpp"
#include "image/ImageGenerator.hpp"
#include "image/ImageWriter.hpp"
#include "neuralnetwork/Network.hpp"
#include "util/Timer.hpp"
#include "util/Util.hpp"

using namespace std;
using Eigen::MatrixXd;

static constexpr unsigned NUM_DERIVED_IMAGES = 5;

static constexpr float GENERATED_IMAGE_SHIFT_X = 0.1f;
static constexpr float GENERATED_IMAGE_SHIFT_Y = 0.1f;
static constexpr float GENERATED_IMAGE_ROT_THETA = 15.0f * M_PI / 180.0f;

static const ImageGenerator imageGenerator(GENERATED_IMAGE_SHIFT_X, GENERATED_IMAGE_SHIFT_Y,
                                           GENERATED_IMAGE_ROT_THETA);

map<int, vector<CharImage>> loadLabeledImages(string imagePath, string labelPath) {
  IdxImages imageLoader(imagePath);
  IdxLabels labelLoader(labelPath);

  vector<int> labels = labelLoader.Load();
  vector<CharImage> images = imageLoader.Load();

  assert(labels.size() == images.size());

  map<int, vector<CharImage>> result;
  for (unsigned i = 0; i < labels.size(); i++) {
    if (result.find(labels[i]) == result.end()) {
      result[labels[i]] = vector<CharImage>();
    }

    result[labels[i]].push_back(images[i]);
  }

  return result;
}

map<int, vector<CharImage>> generateDerivedImages(const map<int, vector<CharImage>> &labeledImages,
                                                  string outDirectory, unsigned numDerived) {

  assert(numDerived >= 1);
  map<int, vector<CharImage>> result;

  ImageWriter writer;

  for (const auto &entry : labeledImages) {
    int digit = entry.first;
    int index = 0;

    result[digit] = vector<CharImage>();
    result[digit].reserve(entry.second.size() * numDerived);

    for (const auto &image : entry.second) {
      vector<CharImage> generated = imageGenerator.GenerateImages(image, numDerived);

      for (auto &gimage : generated) {
        assert(gimage.width == image.width && gimage.height == image.height);
        result[digit].push_back(gimage);
        //
        // stringstream filename;
        // filename << outDirectory << digit << "_" << (index++) << ".png";
        //
        // writer.WriteImage(gimage, filename.str());
      }
      // break;
    }
  }

  return result;
}

TrainingSample sampleFromCharImage(int label, const CharImage &img) {
  Vector output(10);
  // probably there is a better way to do this.
  for (unsigned i = 0; i < 10; i++) {
    output(i) = 0.0f;
  }
  output[label] = 1.0f;

  Vector input(img.pixels.size());
  for (unsigned i = 0; i < img.pixels.size(); i++) {
    input(i) = img.pixels[i];
  }

  return TrainingSample(input, output);
}

vector<TrainingSample> loadSamples(string inImagePath, string inLabelPath, bool genDerived) {
  auto labeledImages = loadLabeledImages(inImagePath, inLabelPath);

  if (genDerived) {
    labeledImages = generateDerivedImages(labeledImages, "data/images/", NUM_DERIVED_IMAGES);
  }

  vector<TrainingSample> result;
  int inputSize = 0;
  int outputSize = 0;

  for (const auto &entry : labeledImages) {
    for (const auto &image : entry.second) {
      result.push_back(sampleFromCharImage(entry.first, image));
      inputSize = result.back().input.rows();
      outputSize = result.back().expectedOutput.rows();
    }
  }

  for (const auto &sample : result) {
    assert(inputSize > 0 && outputSize > 0);
    assert(sample.input.rows() == inputSize);
    assert(sample.expectedOutput.rows() == outputSize);
  }

  return result;
}

int digitFromNNOutput(const Vector &out) {
  assert(out.rows() == 10);

  int result = 0;
  float maxActivation = out(0);

  for (int i = 1; i < out.rows(); i++) {
    if (out(i) > maxActivation) {
      maxActivation = out(i);
      result = i;
    }
  }

  return result;
}

float testNetwork(Network &network, const std::vector<TrainingSample> &evalSamples) {
  unsigned numCorrect = 0;

  for (const auto &es : evalSamples) {
    auto result = network.Process(es.input);
    // cout << es << " -> " << result << endl << endl;

    bool isCorrect = digitFromNNOutput(result) == digitFromNNOutput(es.expectedOutput);
    numCorrect += isCorrect ? 1 : 0;
  }

  return 1.0f - (numCorrect / static_cast<float>(evalSamples.size()));
}

int main() {
  Eigen::initParallel();
  srand(1234);

  cout << "loading training data" << endl;
  vector<TrainingSample> trainingSamples =
      loadSamples("data/train_images.idx3", "data/train_labels.idx1", true);
  random_shuffle(trainingSamples.begin(), trainingSamples.end());
  cout << "training data size: " << trainingSamples.size() << endl;

  cout << "loading test data" << endl;
  vector<TrainingSample> testSamples =
      loadSamples("data/test_images.idx3", "data/test_labels.idx1", false);
  random_shuffle(testSamples.begin(), testSamples.end());
  cout << "test data size: " << testSamples.size() << endl;

  unsigned inputSize = trainingSamples.front().input.rows();
  unsigned outputSize = trainingSamples.front().expectedOutput.rows();

  Network network({inputSize, inputSize, outputSize});
  // uptr<Trainer> trainer = make_unique<SimpleTrainer>(0.2, 0.001, 500);
  uptr<Trainer> trainer = make_unique<DynamicTrainer>(0.5f, 0.00001f, 0.5f, 0.25f, 1000, 30000);

  trainer->AddProgressCallback(
      [&trainingSamples, &testSamples](Network &network, float trainError, unsigned iter) {
        if (iter % 100 == 0) {
          float testWrong = testNetwork(network, testSamples);
          float trainWrong = testNetwork(network, trainingSamples);
          cout << iter << "\t" << trainError << "\t" << testWrong << "\t" << trainWrong << endl;
        }
      });

  cout << "starting training..." << endl;
  trainer->Train(network, trainingSamples, 100000);

  cout << "finished" << endl;
  return 0;
}
