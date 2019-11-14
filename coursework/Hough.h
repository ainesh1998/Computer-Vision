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
    Mat hough(Mat &image, int centreX, int centreY, int radius);
    int addR(int x, int y);
};

#endif
