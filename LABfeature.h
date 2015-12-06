#ifndef _720PROJECT_LABFEATURE_H
#define _720PROJECT_LABFEATURE_H

#include <opencv2/opencv.hpp>
#include "feature.h"
using namespace cv;

class LABFeature : public Feature {
 public:
  virtual void resetImage(Mat img) override {
    Mat imgLab;
    cvtColor(img, imgLab, CV_BGR2Lab);
    Mat means = Mat::ones(3, 1, CV_32F)*128;
    _img = dstretch(imgLab, means);
  }
  // Evaluates the given feature value at linear pixel id
  virtual Mat evaluate(int id) override;
};

#endif // _720PROJECT_LABFEATURE_H
