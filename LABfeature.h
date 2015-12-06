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
    _img = dstretch(imgLab);
  }
  // Evaluates the given feature value at linear pixel id
  virtual Mat evaluate(int id) override;
};

#endif // _720PROJECT_LABFEATURE_H
