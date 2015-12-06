#include "LABfeature.h"
using namespace cv;

Mat LABFeature::evaluate(int id) {
  int row = id / _img.cols, col = id % _img.cols;
  Vec3d val = _img.at<Vec3d>(row, col) + Vec3d{0, 127, 127};
  Mat res = (Mat(val) / Mat(Vec3d{100, 254, 254}));
  return res.reshape(0, 1);
}
