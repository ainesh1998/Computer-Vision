#ifndef HOUGH_HELPER
#define HOUGH_HELPER
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

class HoughHelper{
public:
     void sobel(Mat &gray_image, Mat &mag_image, Mat &dir_image);
     Mat threshold(Mat &image, int threshVal);
     Mat thresholdHough(string img, int threshVal);
     void overlayHough(Mat &original, Mat &hough_centres);

};

#endif
