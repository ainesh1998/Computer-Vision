#ifndef HOUGH_LINE_TRANSFORM
#define HOUGH_LINE_TRANSFORM
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

class HoughLine {
public:
    Mat line_detect(Mat &image);
};

#endif
