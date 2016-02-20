
#include "Tensor.hpp"
#include <cassert>
#include <algorithm>


unsigned Tensor::NumLayers(void) const {
  return this->data.size();
}

void Tensor::AddLayer(const Matrix &m) {
  this->data.push_back(m);
}

Matrix& Tensor::operator()(unsigned index) {
    assert(index < data.size());
    return data[index];
}

const Matrix& Tensor::operator()(unsigned index) const {
  assert(index < data.size());
  return data[index];
}

Tensor Tensor::operator+(const Tensor &t) const {
  Tensor result(*this);
  result += t;
  return result;
}

Tensor Tensor::operator-(const Tensor &t) const {
  Tensor result(*this);
  result -= t;
  return result;
}

Tensor Tensor::operator*(float s) const {
  Tensor result(*this);
  result *= s;
  return result;
}

Tensor Tensor::operator/(float s) const {
  Tensor result(*this);
  result /= s;
  return result;
}

Tensor& Tensor::operator+=(const Tensor &t) {
  assert(this->NumLayers() == t.NumLayers());
  for (unsigned i = 0; i < NumLayers(); i++) {
    data[i] += t.data[i];
  }
  return *this;
}

Tensor& Tensor::operator-=(const Tensor &t) {
  assert(this->NumLayers() == t.NumLayers());
  for (unsigned i = 0; i < NumLayers(); i++) {
    data[i] -= t.data[i];
  }
  return *this;
}

Tensor& Tensor::operator*=(float s) {
  for_each(data, [=] (Matrix &m) { m *= s; });
  return *this;
}

Tensor& Tensor::operator/=(float s) {
  float inv = 1.0f / s;
  for_each(data, [=] (Matrix &m) { m *= inv; });
  return *this;
}
