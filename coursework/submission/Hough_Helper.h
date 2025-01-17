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
     Mat sobel(Mat &gray_image);
     Mat threshold(string input, string output, int threshVal);
     void overlayHough(Mat &original, Mat &hough_centres, string name);

};

#endif
