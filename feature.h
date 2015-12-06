#ifndef _720PROJECT_FEATURE_H
#define _720PROJECT_FEATURE_H

#include <opencv2/opencv.hpp>
using namespace cv;

class Feature {
 public:
  // Resets/preprocesses the feature evaluator to work on img
  virtual void resetImage(Mat img) {
    _img = img;
  }
  // Evaluates the given feature value at linear pixel id
  virtual Mat evaluate(int id) = 0;
 protected:
  Mat _img;
};

#endif // _720PROJECT_FEATURE_H
