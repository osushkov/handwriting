#pragma once

#include "../common/Common.hpp"
#include "CharImage.hpp"

class ImageWriter {
public:
  void writeImage(const CharImage &img, string outPath);
};
