#include <iostream>
#include <cassert>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;

const int lowThreshold = 55;
const int ratio = 3;
Mat ProcessSingleImage(Mat src) {
    assert(src.data);

    Mat edges; cvtColor(src, edges, CV_BGR2GRAY);
    blur(edges, edges, Size(3,3));
    Canny(edges, edges, lowThreshold, lowThreshold*ratio, 3);
    return edges;
}
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
