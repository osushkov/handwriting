
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
#include "util/Util.hpp"

using namespace std;
using Eigen::MatrixXd;

map<int, vector<CharImage>> loadLabeledImages(string imagePath,
                                              string labelPath) {
  IdxImages imageLoader(imagePath);
  IdxLabels labelLoader(labelPath);

  vector<int> labels = labelLoader.load();
  vector<CharImage> images = imageLoader.load();

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

map<int, vector<CharImage>> generateDerivedImages(string inImagePath,
                                                  string inLabelPath,
                                                  string outDirectory,
                                                  unsigned numDerived) {
  assert(numDerived >= 1);
  map<int, vector<CharImage>> result;

  ImageWriter writer;
  ImageGenerator generator;

  auto labeledImages = loadLabeledImages(inImagePath, inLabelPath);
  for (auto &entry : labeledImages) {
    int digit = entry.first;
    int index = 0;

    result[digit] = vector<CharImage>();
    result[digit].reserve(entry.second.size() * numDerived);

    for (auto &image : entry.second) {
      vector<CharImage> generated = generator.GenerateImages(image, numDerived);

      for (auto &gimage : generated) {
        assert(gimage.width == image.width && gimage.height == image.height);
        result[digit].push_back(gimage);

        stringstream filename;
        filename << outDirectory << digit << "_" << (index++) << ".png";

        writer.writeImage(gimage, filename.str());
      }
      break;
    }
  }

  return result;
}

void evaluateNetwork(Network &network,
                     const std::vector<TrainingSample> &evalSamples) {
  unsigned numCorrect = 0;

  for (const auto &es : evalSamples) {
    auto result = network.Process(es.input);
    // cout << es << " -> " << result << endl << endl;

    for (unsigned i = 0; i < result.rows(); i++) {
      bool isCorrect = (result(i) > 0.6f && es.expectedOutput(i) > 0.5f) ||
                       (result(i) < 0.4f && es.expectedOutput(i) < 0.5f);
      numCorrect += isCorrect ? 1 : 0;
    }
  }

  cout << "frac correct: " << (numCorrect / (float)evalSamples.size()) << endl;
}

int main() {
  // IdxImages idxImages("data/train_images.idx3");
  // idxImages.load();

  generateDerivedImages("data/test_images.idx3", "data/test_labels.idx1",
                        "data/images/", 10);

  /*
    srand(1234);

    Network network({2, 3, 1});
    // uptr<Trainer> trainer = make_unique<SimpleTrainer>(0.2, 0.001, 500);
    uptr<Trainer> trainer = make_unique<DynamicTrainer>(0.5f, 0.5f, 0.25f, 500);

    vector<TrainingSample> trainingSamples = getTrainingData(8000);
    trainer->Train(network, trainingSamples, 100000);

    vector<TrainingSample> evalSamples = getTrainingData(1000);
    evaluateNetwork(network, evalSamples);
  */
  cout << "finished" << endl;
  getchar();
  return 0;
}
