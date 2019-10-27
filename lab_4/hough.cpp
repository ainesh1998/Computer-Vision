// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#define THRESHOLD 127
using namespace cv;
using namespace std;

Mat threshold(Mat &image) {
    Mat thr(image.rows,image.cols,CV_8UC1,Scalar(0));
    for(int y = 0; y < image.rows; y++){
        for(int x = 0; x < image.cols; x++){
            if(image.at<uchar>(y,x) > THRESHOLD){
                thr.at<uchar>(y,x) = 255;
            }else{
                thr.at<uchar>(y,x) = 0;
            }
        }
    }
    imwrite("thr.jpg",thr);
    return thr;
}
int main(int argc, char const *argv[]) {
    Mat image = imread("mag.jpg",0);
    threshold(image);
    return 0;
}
