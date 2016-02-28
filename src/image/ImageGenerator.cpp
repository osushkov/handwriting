
#include "ImageGenerator.hpp"
#include "../util/Util.hpp"

#include <cmath>
#include <opencv2/opencv.hpp>

using Vec1b = cv::Vec<unsigned char, 1>;

vector<CharImage> ImageGenerator::GenerateImages(const CharImage &base,
                                                 unsigned numImages) {

  cv::Mat cvImg = convertToMat(base);

  vector<CharImage> result;
  result.reserve(numImages);

  result.push_back(base);
  for (unsigned i = 0; i < numImages - 1; i++) {
    auto t = randomTransform(base);
    result.push_back(transformToCharImage(cvImg, t));
  }

  return result;
}

ImageGenerator::Transform
ImageGenerator::randomTransform(const CharImage &img) const {
  const double WIDTH_FRAC_OFFSET = 0.25;
  const double HEIGHT_FRAC_OFFSET = 0.25;
  const double ROT_THETA = 45.0 * M_PI / 180.0;

  return Transform(
      img.width * Util::RandInterval(-WIDTH_FRAC_OFFSET, WIDTH_FRAC_OFFSET),
      img.height * Util::RandInterval(-HEIGHT_FRAC_OFFSET, HEIGHT_FRAC_OFFSET),
      Util::RandInterval(-ROT_THETA, ROT_THETA));
}

cv::Mat ImageGenerator::convertToMat(const CharImage &img) const {
  cv::Mat outImg(img.width, img.height, CV_8UC1);

  for (int y = 0; y < outImg.rows; y++) {
    for (int x = 0; x < outImg.cols; x++) {
      Vec1b &v = outImg.at<Vec1b>(y, x);
      v[0] = static_cast<unsigned char>(img.pixels[x + y * img.width] * 255);
    }
  }

  return outImg;
}

CharImage ImageGenerator::convertToCharImage(const cv::Mat &img) const {
  vector<float> pixels;
  pixels.reserve(img.cols * img.rows);

  for (int y = 0; y < img.rows; y++) {
    for (int x = 0; x < img.cols; x++) {
      const Vec1b &v = img.at<Vec1b>(y, x);
      pixels.push_back(static_cast<float>(v[0]) / 255.0f);
    }
  }

  return CharImage(img.cols, img.rows, pixels);
}

CharImage ImageGenerator::transformToCharImage(
    const cv::Mat &src, const ImageGenerator::Transform &transform) const {

  cv::Mat rot =
      cv::getRotationMatrix2D(cv::Point(src.rows / 2.0, src.cols / 2.0),
                              transform.theta * 180.0 / M_PI, 1.0);
  cv::Mat rotated; // = cv::Mat::zeros(src.rows, src.cols, src.type());
  warpAffine(src, rotated, rot, src.size());
  // cv::transform(src, rotated, rot);

  // return convertToCharImage(rotated);

  cv::Mat trans = (cv::Mat_<float>(2, 3) << 1.0f, 0.0f, transform.tx, 0.0f,
                   1.0f, transform.ty);
  cv::Mat translated; // = cv::Mat::zeros(src.rows, src.cols, src.type());
  warpAffine(rotated, translated, trans, src.size());
  // cv::transform(rotated, translated, trans);

  return convertToCharImage(translated);
}
