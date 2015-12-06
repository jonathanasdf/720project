#include <iostream>
#include <cassert>
#include <vector>
#include <memory>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "globalPb.h"
#include "smatrix.h"

#include "feature.h"
#include "LABfeature.h"

using namespace cv;
using namespace std;

/*** Global variables ***/
const double PI = acos(-1.);
// Number of pairs of pixels to sample for density estimation
const int NUM_SAMPLES = 10000;
RNG rng;
// Feature evaluation functions
vector<shared_ptr<Feature> > features = {
  shared_ptr<Feature>(new LABFeature())
};
// Vector for bandwidth for Epanechnikov kernel for each filter
vector<Mat> bandwidth = {
  Mat(Vec3d{0.1, 0.01, 0.01})
};



/*** For sampling pairs of pixels inversely weighted by distance ***/
pair<int, int> generateOneSample(int nrows, int ncols) {
    int row1 = -1, col1 = -1, row2 = -1, col2 = -1;
    while (row2 < 0 || col2 < 0 || row2 >= nrows || col2 >= ncols || (row1 == row2 && col1 == col2)) {
        row1 = rng.uniform(0, nrows);
        col1 = rng.uniform(0, ncols);
        double t = rng.uniform(0., 2*PI);
        double d = 1 + abs(rng.gaussian(1));
        row2 = round(row1 + cos(t)*d);
        col2 = round(col1 + sin(t)*d);
    }
    return make_pair(row1 * ncols + col1, row2 * ncols + col2);
}
vector<pair<int, int> > generateSamples(int nrows, int ncols, int nsamples) {
    vector<pair<int, int> > res;
    for (int i=0; i < nsamples; i++)
      res.push_back(generateOneSample(nrows, ncols));
    return res;
}



/*** Find edges for a single image ***/
Mat ProcessSingleImage(Mat src) {
    assert(src.data);

    // sample pairs of pixels
    vector<pair<int, int> > samples = generateSamples(src.rows, src.cols, NUM_SAMPLES);

    // convert pixels into feature space vector
    vector<vector<Mat> > sampledFeatures;
    for (int i=0; i < features.size(); i++) {
      features[i]->resetImage(src);
      vector<Mat> converted;
      for (int j=0; j < samples.size(); j++) {
        Mat f1 = features[i]->evaluate(samples[j].first),
            f2 = features[i]->evaluate(samples[j].second);
        Mat c1, c2;
        hconcat(f1, f2, c1);
        hconcat(f2, f1, c2);
        converted.push_back(c1);
        converted.push_back(c2);
      }
      sampledFeatures.push_back(converted);
    }

    // compute affinity matrix

    return src;
}


/*** Entry point and handles batch processing. ***/
int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: " << argv[0]
             << " <image_paths> [output_folder (required if >1 image)]" << endl;
        return -1;
    }

    if (argc == 2) {
        Mat image = imread(argv[1], 1);
        namedWindow("Original Image", WINDOW_AUTOSIZE);
        imshow("Original Image", image);

        Mat processed = ProcessSingleImage(image);
        namedWindow("Processed Image", WINDOW_AUTOSIZE);
        imshow("Processed Image", processed);
    } else {
        string output_dir(argv[argc-1]);
        if (string("/\\").find(output_dir[output_dir.size()-1]) == -1) {
            output_dir += "/";
        }
        boost::filesystem::create_directories(output_dir);
        for (int i=1; i < argc-1; i++) {
            string filename(argv[i]);
            cout << "Processing " << filename << endl;
            Mat image = imread(argv[i], 1);
            Mat processed = ProcessSingleImage(image);
            imwrite(output_dir + filename.substr(filename.find_last_of("/\\")+1),
                    processed);
        }
    }
    waitKey(0);
    return 0;
}
