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
const double EPS = 1e-9;
// Number of pairs of pixels to sample for density estimation
const int NUM_SAMPLES = 10000;
RNG rng;
// Feature evaluation functions
const vector<shared_ptr<Feature> > features = {
  shared_ptr<Feature>(new LABFeature())
};
// Vector for bandwidth for Epanechnikov kernel for each filter
const vector<Mat> bandwidth = {
  Mat(Vec3d{0.1, 0.01, 0.01}).t()
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



/*** Computation of PMI ***/
double evaluateP(const Mat &x, flann::Index &index, const vector<Mat> &samples, const Mat& bandwidth) {
  Mat_<int> indices;
  Mat_<float> dists;
  double radius;
  minMaxIdx(bandwidth, NULL, &radius);
  index.radiusSearch(x, indices, dists, radius, samples.size());

  double res = 0;
  for (int i=0; i < indices.rows; i++) {
    Mat d = ((x - samples[indices(i, 0)]) / bandwidth);
    res += max(0., (1 - d.dot(d))*3/4);
  }
  return res;
}
vector<pair<Mat, double> > marginalCache;
double computeMarginal(const Mat &f, const vector<Mat> &samples, const Mat& bandwidth) {
  for (int i=0; i < marginalCache.size(); i++) {
    if (norm(marginalCache[i].first, f) < EPS) {
      return marginalCache[i].second;;
    }
  }

  double res = 0;
  Mat bw1 = bandwidth(Range::all(), Range(1, f.cols)),
      bw2 = bandwidth(Range::all(), Range(f.cols, 2*f.cols));
  for (int i = 0; i < samples.size(); i++) {
    Mat a = samples[i](Range::all(), Range(1, f.cols)),
        b = samples[i](Range::all(), Range(f.cols, 2*f.cols));
    Mat d = (f - a) / bw1,
        o = (Mat::ones(1, f.cols, CV_64F) - b) / bw2,
        z = (Mat::zeros(1, f.cols, CV_64F) - b) / bw2;
    res += max(0., (1 - d.dot(d) - (o.mul(o).mul(o) - z.mul(z).mul(z)).dot(bw2)/3)*3/4);
  }

  marginalCache.push_back(make_pair(f, res));
  return res;
}
const double rho = 1.25;
const double regularizer = 100;
double computePMI(const Mat &f1, const Mat &f2, flann::Index &index, const vector<Mat> &samples, const Mat &bandwidth) {
  Mat x; hconcat(f1, f2, x);
  return (pow(evaluateP(x, index, samples, bandwidth), rho) + regularizer)/(computeMarginal(f1, samples, bandwidth)*computeMarginal(f2, samples, bandwidth) + regularizer);
}



/*** Calculation of affinity matrix ***/
const int window = 5;
SMatrix* calculateAffinityMatrix(const Mat &src, const vector<vector<Mat> > &sampledFeatures, vector<flann::Index> &indexes) {
  marginalCache.clear();

  int numPixels = src.rows * src.cols;
  int* nz = new int[numPixels];
  int** cols = new int*[numPixels];
  double** vals = new double*[numPixels];

  vector<Mat> f[numPixels];
  for (int i=0; i < numPixels; i++) {
    for (int j=0; j < features.size(); j++) {
      Mat ff = features[j]->evaluate(i);
      Mat check; inRange(ff, 0, 1, check);
      assert(countNonZero(check) == ff.total());
      f[i].push_back(ff);
    }
  }

  double mn = 1e18, mx = -1e18;
  for (int i=0; i < numPixels; i++) {
    nz[i] = 0;
    vector<int> col;
    vector<double> val;
    int r = i / src.cols, c = i % src.cols;
    for (int u = -window; u <= window; u++) {
      int rr = r + u;
      if (rr < 0 || rr >= src.rows) continue;
      for (int v = -window; v <= window; v++) {
        int cc = c + v;
        if (cc < 0 || cc >= src.cols) continue;
        if (u*u+v*v > window*window) continue;

        double pmi = 1;
        for (int j=0; j < features.size(); j++) {
          pmi *= computePMI(f[i][j], f[rr*src.cols+cc][j], indexes[j], sampledFeatures[j], bandwidth[j]);
        }
        mn = min(mn, pmi);
        mx = max(mx, pmi);

        nz[i]++;
        col.push_back(rr*src.cols + cc);
        val.push_back(pmi);
      }
    }
    cols[i] = new int[nz[i]];
    vals[i] = new double[nz[i]];
    for (int j=0; j < nz[i]; j++) {
      cols[i][j] = col[j];
      vals[i][j] = val[j];
    }
  }

  for (int i=0; i < numPixels; i++) {
    for (int j=0; j < nz[i]; j++) {
      vals[i][j] = (vals[i][j] + mn) / (mx + mn);
      assert(vals[i][j] >= 0 && vals[i][j] <= 1);
    }
  }

  return new SMatrix(numPixels, nz, cols, vals);
}



/*** Find edges for a single image ***/
Mat ProcessSingleImage(const Mat &src, bool debug = false) {
  assert(src.data);

  // sample pairs of pixels
  if (debug) cout << "Sampling " << NUM_SAMPLES << " pairs of pixel locations..." << endl;
  vector<pair<int, int> > samples = generateSamples(src.rows, src.cols, NUM_SAMPLES);

  // convert pixels into feature space vector
  if (debug) cout << "Converting pixel pairs into feature space vectors..." << endl;
  vector<vector<Mat> > sampledFeatures;
  vector<flann::Index> indexes;
  for (int i=0; i < features.size(); i++) {
    features[i]->resetImage(src);
    vector<Mat> converted;
    for (int j=0; j < samples.size(); j++) {
      Mat f1 = features[i]->evaluate(samples[j].first),
          f2 = features[i]->evaluate(samples[j].second);
      Mat x1, x2;
      hconcat(f1, f2, x1);
      hconcat(f2, f1, x2);
      converted.push_back(x1);
      converted.push_back(x2);
    }
    sampledFeatures.push_back(converted);
    Mat data; vconcat(converted, data);
    indexes.push_back(flann::Index(data, flann::KDTreeIndexParams()));
  }

  // calculate affinity matrix
  if (debug) cout << "Calculating affinity matrix..." << endl;
  SMatrix* W = calculateAffinityMatrix(src, sampledFeatures, indexes);

  // run spectral clustering via gPb
  Mat gPb, gPb_thin;
  vector<Mat> gPb_ori;
  globalPb(src, W, gPb, gPb_thin, gPb_ori);
  return gPb;
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

    Mat processed = ProcessSingleImage(image, true);
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
      imwrite(output_dir + filename.substr(filename.find_last_of("/\\")+1), processed);
    }
  }
  waitKey(0);
  return 0;
}
