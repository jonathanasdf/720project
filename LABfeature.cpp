#include <iostream>
#include "LABfeature.h"
using namespace cv;

Mat LABFeature::evaluate(int id) {
  int row = id / _img.cols, col = id % _img.cols;
  Vec3f val = _img.at<Vec3f>(row, col);
  Mat res = (Mat(val) + Mat(Vec3f(0, 127, 127))) / Mat(Vec3f(100, 255, 255));
  return min(res.reshape(0, 1), 1);
}
