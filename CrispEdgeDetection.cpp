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
const int NUM_SAMPLES = 1000;
RNG rng;
// Feature evaluation functions
const vector<shared_ptr<Feature> > features = {
  shared_ptr<Feature>(new LABFeature())
};
// Vector for bandwidth for Epanechnikov kernel for each filter
const vector<Mat> bandwidth = {
  Mat(Vec3f{0.1, 0.05, 0.05}).t()
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
double evaluateP(const Mat &x, const Mat &samples, flann::Index *kdTree, const Mat& bandwidth) {
  double radius; minMaxIdx(bandwidth, NULL, &radius);
  vector<int> indices;
  vector<float> dists;
  kdTree->radiusSearch(x, indices, dists, radius*radius, samples.rows);

  Mat bw; hconcat(bandwidth, bandwidth, bw);

  double res = 0;
  for (int i=0; i < indices.size(); i++) {
    if (i != 0 && dists[i] == 0) break;
    Mat d = ((x - samples.row(indices[i]))) / bw;
    res += max(0., (1 - d.dot(d))*3/4);
  }
  return res;
}

map<int, double> marginalCache;
double computeMarginal(const vector<Mat> &f, int p, const Mat &samples, const Mat& bandwidth) {
  if (marginalCache.count(p)) return marginalCache[p];

  double res = 0;
  for (int i = 0; i < samples.rows; i++) {
    Mat a = samples(Range(i, i+1), Range(0, f[p].cols)),
        b = samples(Range(i, i+1), Range(f[p].cols, 2*f[p].cols));
    Mat d = (f[p] - a) / bandwidth;
    double mx; minMaxIdx(d, NULL, &mx);
    if (mx > 1) continue;

    Mat o = (Mat::ones(1, f[p].cols, CV_32F) - b) / bandwidth,
        z = (Mat::zeros(1, f[p].cols, CV_32F) - b) / bandwidth;
    res += max(0., (1 - d.dot(d) - (o.mul(o).mul(o) - z.mul(z).mul(z)).dot(bandwidth)/3)*3/4);
  }

  return marginalCache[p] = res;
}

const double rho = 1.25;
const double regularizer = 1e-3;
double computePMI(const vector<Mat> &f, int p1, int p2, const Mat &samples, flann::Index *kdTree, const Mat &bandwidth) {
  Mat x; hconcat(f[p1], f[p2], x);
  double P12 = evaluateP(x, samples, kdTree, bandwidth),
         P1 = computeMarginal(f, p1, samples, bandwidth),
         P2 = computeMarginal(f, p2, samples, bandwidth);
  return (pow(P12, rho) + regularizer)/(P1*P2 + regularizer);
}



/*** Calculation of affinity matrix ***/
const int window = 5;
SMatrix* calculateAffinityMatrix(const Mat &src, const vector<Mat> &sampledFeatures, vector<flann::Index*> kdTrees, bool debug) {
  int numPixels = src.rows * src.cols;
  int* nz = new int[numPixels];
  int** cols = new int*[numPixels];
  double** vals = new double*[numPixels];

  vector<vector<Mat> > f(features.size(), vector<Mat>(numPixels));
  for (int i=0; i < features.size(); i++) {
    for (int j=0; j < numPixels; j++) {
      f[i][j] = features[i]->evaluate(j);
      Mat check; inRange(f[i][j], 0, 1, check);
      assert(countNonZero(check) == f[i][j].total());
    }
  }

  marginalCache.clear();
  unordered_map<long long, double> pmiCache;

  double mn = 1e18, mx = -1e18;
  for (int i=0; i < numPixels; i++) {
    nz[i] = 0;
    vector<int> col;
    vector<double> val;
    int r = i / src.cols, c = i % src.cols;
    if (debug) if (c%10 == 0) cout << "Computing PMI values for pixel (" << r << "," << c << ")" << endl;

    for (int u = -window; u <= window; u++) {
      int rr = r + u;
      if (rr < 0 || rr >= src.rows) continue;
      for (int v = -window; v <= window; v++) {
        int cc = c + v;
        if (cc < 0 || cc >= src.cols) continue;
        if (u*u+v*v > window*window) continue;

        int ii = rr*src.cols+cc;
        int key = ((long long)min(i, ii)) * numPixels + max(i, ii);

        double pmi = 1;
        if (i > ii) {
          pmi = pmiCache[key];
        } else {
          for (int j=0; j < features.size(); j++) {
            pmi *= computePMI(f[j], i, ii, sampledFeatures[j], kdTrees[j], bandwidth[j]);
          }
          pmiCache[key] = pmi;
          mn = min(mn, pmi);
          mx = max(mx, pmi);
        }

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

  Mat img;
  src.copyTo(img);
  // downsample image for speed
  resize(img, img, Size(), 0.5, 0.5);

  // sample pairs of pixels
  if (debug) cout << "Sampling " << NUM_SAMPLES << " pairs of pixel locations..." << endl;
  vector<pair<int, int> > samples = generateSamples(img.rows, img.cols, NUM_SAMPLES);

  // convert pixels into feature space vector
  if (debug) cout << "Converting pixel pairs into feature space vectors..." << endl;
  vector<Mat> sampledFeatures;
  vector<flann::Index*> kdTrees;
  for (int i=0; i < features.size(); i++) {
    features[i]->resetImage(img);
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
    Mat data; vconcat(converted, data);
    data.convertTo(data, CV_32F);
    sampledFeatures.push_back(data);
    kdTrees.push_back(new flann::Index(sampledFeatures.back(), flann::KDTreeIndexParams()));
  }

  // calculate affinity matrix
  if (debug) cout << "Calculating affinity matrix..." << endl;
  SMatrix* W = calculateAffinityMatrix(img, sampledFeatures, kdTrees, debug);

  for (int i=0; i < features.size(); i++) {
    delete kdTrees[i];
  }

  // run spectral clustering via gPb
  Mat gPb, gPb_thin;
  vector<Mat> gPb_ori;
  globalPb(img, W, gPb, gPb_thin, gPb_ori);

  if (debug) cout << "Completed." << endl;

  delete W;
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
