
#include <Eigen/Core>
#include <Eigen/Dense>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "DynamicTrainer.hpp"
#include "DynamicTrainerBuilder.hpp"
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

// Number of images to generate using rotation and translation from each canonical training image.
static constexpr unsigned NUM_DERIVED_IMAGES = 1;

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

  for (const auto &entry : labeledImages) {
    int digit = entry.first;

    result[digit] = vector<CharImage>();
    result[digit].reserve(entry.second.size() * numDerived);

    for (const auto &image : entry.second) {
      vector<CharImage> generated = imageGenerator.GenerateImages(image, numDerived);

      for (auto &gimage : generated) {
        assert(gimage.width == image.width && gimage.height == image.height);
        result[digit].push_back(gimage);
      }
    }
  }

  return result;
}

TrainingSample sampleFromCharImage(int label, const CharImage &img) {
  Vector output(10);
  output.fill(0.0f);
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

uptr<Trainer> getTrainer(void) {
  DynamicTrainerBuilder builder;

  builder.StartLearnRate(0.5f)
      .FinishLearnRate(0.001f)
      .MaxLearnRate(0.5f)
      .Momentum(0.25f)
      .StartSamplesPerIter(1000)
      .FinishSamplesPerIter(10000)
      .UseMomentum(true)
      .UseSpeedup(true)
      .UseWeightRates(true);

  return builder.Build();
}

Network createNewNetwork(unsigned inputSize, unsigned outputSize) {
  vector<unsigned> networkLayers = {inputSize, inputSize, outputSize};
  return Network(networkLayers);
}

Network loadNetwork(string path) {
  ifstream networkIn(path, ios::in | ios::binary);
  return Network(networkIn);
}

void learn(Network &network, vector<TrainingSample> &trainingSamples,
           vector<TrainingSample> &testSamples) {
  auto trainer = getTrainer();

  trainer->AddProgressCallback(
      [&trainingSamples, &testSamples](Network &network, float trainError, unsigned iter) {
        if (iter % 10 == 0) {
          float testWrong = testNetwork(network, testSamples);
          cout << iter << "\t" << trainError << "\t" << testWrong << endl;

          // float trainWrong = testNetwork(network, trainingSamples);
          // cout << iter << "\t" << trainError << "\t" << testWrong << "\t" << trainWrong << endl;
        }
      });

  cout << "starting training..." << endl;
  trainer->Train(network, trainingSamples, 100);
  cout << "finished" << endl;
}

void eval(Network &network, const vector<TrainingSample> &testSamples) {
  unsigned numCorrect = 0;

  for (const auto &ts : testSamples) {
    auto result = network.Process(ts.input);
    bool isCorrect = digitFromNNOutput(result) == digitFromNNOutput(ts.expectedOutput);
    numCorrect += isCorrect ? 1 : 0;
  }

  cout << endl
       << "percent correct: " << (100.0f * (numCorrect / static_cast<float>(testSamples.size())))
       << endl;
  ;
}

int main(int argc, char **argv) {
  Eigen::initParallel();
  srand(1234);

  // TODO: training+test image data paths can be command line args.

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

  // TODO: should probably use a command line args parsing library here.
  if (argc == 1 || (argc >= 2 && string(argv[1]) == "train")) {
    Network network = argc == 3 ? loadNetwork(argv[2]) : createNewNetwork(inputSize, outputSize);
    learn(network, trainingSamples, testSamples);

    ofstream networkOut("network.dat", ios::out | ios::binary);
    network.Serialize(networkOut);
  } else if (argc == 3 && string(argv[1]) == "eval") {
    Network network = loadNetwork(argv[2]);
    eval(network, testSamples);
  } else {
    cout << "invalid arguments, expected: " << endl;
    cout << string(argv[0]) << " train [existing_network_file]" << endl;
    cout << string(argv[0]) << " test network_file" << endl;
  }

  return 0;
}
