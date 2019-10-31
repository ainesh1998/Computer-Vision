// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
// #include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define THRESHOLD 127
#define MAX_RADIUS 100
#define MIN_RADIUS 20


int ***malloc3dArray(int centreX, int centreY, int radius){
    int i, j, k;
    int ***array = (int ***) malloc(centreX * sizeof(int **));

    for (i = 0; i < centreX; i++) {
        array[i] = (int **) malloc(centreY * sizeof(int *));
        for (j = 0; j < centreY; j++) {
            array[i][j] = (int *) malloc(radius * sizeof(int));
        }
    }
    return array;
}

Mat threshold(Mat &image) {
    Mat thr(image.rows,image.cols,CV_8UC1,Scalar(0));
    for(int y = 0; y < image.rows; y++){
        for(int x = 0; x < image.cols; x++){
            if(image.at<uchar>(y,x) > THRESHOLD){
                thr.at<uchar>(y,x) = 255;
            }
            else{
                thr.at<uchar>(y,x) = 0;
            }
        }
    }
    imwrite("thr.jpg",thr);
    return thr;
}

void hough(Mat &thr, Mat &dir, int ***hough_space, int centreX, int centreY, int radius) {
    //zero hough space
    for(int i = 0; i < centreX; i++){
        for(int j = 0; j < centreY; j++){
            for(int k = 0; k < radius; k++){
                hough_space[i][j][k] = 0;
            }
        }
    }

    for(int y = 0; y < thr.rows; y++){
        for(int x = 0; x < thr.cols; x++){
            if(thr.at<uchar>(y,x) == 255){
                //hough voting
                for(int r = 0; r < MAX_RADIUS -MIN_RADIUS; r++){
                    
                }
            }
        }
    }
}

int main(int argc, char const *argv[]) {
    int centreX = 500, centreY = 500, radius =  MAX_RADIUS - MIN_RADIUS;
    int ***hough_space;

    FileStorage fs("dir.yml", FileStorage::READ);
    Mat dir;
    fs["tempDir"] >> dir;

    hough_space = malloc3dArray(centreX, centreY, radius);
    Mat mag = imread("mag.jpg",0);
    Mat thr = threshold(mag);

    hough(thr, dir, hough_space, centreX, centreY, radius);
    return 0;
}
