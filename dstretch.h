#ifndef _720PROJECT_DSTRETCH_H
#define _720PROJECT_DSTRETCH_H
#include <opencv2/opencv.hpp>
using namespace cv;

/*
Source      : http://dhanushkadangampola.blogspot.com/2015/02/decorrelation-stretching.html
input       : p x q x n multi-channel image.
targetMean  : n x 1 vector containing desired mean for each channel of the dstretched image. If empty, mean of the input data is used.
targetSigma : n x 1 vector containing desired sigma for each channel of the dstretched image. If empty, sigma of the input data is used.

returns floating point dstretched image
*/
Mat dstretch(Mat& input);
Mat dstretch(Mat& input, Mat& targetMean, Mat& targetSigma);

#endif // _720PROJECT_DSTRETCH_H
