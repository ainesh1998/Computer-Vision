#ifndef HOUGH_TRANSFORM
#define HOUGH_TRANSFORM
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

class Hough{
public:
    Mat circle_detect(Mat &image);
    void line_detect(Mat &image);
};

#endif
