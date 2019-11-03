// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h>
// #include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define THRESHOLD 80
#define THRESHOLD_HOUGH 80
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

Mat threshold(Mat &image, int threshVal) {
    Mat thr(image.rows,image.cols,CV_8UC1,Scalar(0));
    // std::cout << image.rows << '\n';
    // std::cout << image.cols << '\n';
    for(int y = 0; y < image.rows; y++){
        for(int x = 0; x < image.cols; x++){
            if(image.at<uchar>(y,x) > threshVal){
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

Mat thresholdHough(int threshVal) {
    Mat hough2D = imread("hough_space.jpg", 0);
    Mat thr(hough2D.rows, hough2D.cols,CV_8UC1,Scalar(0));
    for(int y = 0; y < hough2D.rows; y++){
        for(int x = 0; x < hough2D.cols; x++){
            if(hough2D.at<uchar>(y,x) > threshVal){
                thr.at<uchar>(y,x) = 255;
            }
            else{
                thr.at<uchar>(y,x) = 0;
            }
        }
    }
    imwrite("hough_centres.jpg",thr);
    return thr;
}

Mat hough(Mat &thr, Mat &dir, int ***hough_space, int centreX, int centreY, int radius) {
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
                for(int r = 0; r < radius/2; r++) {
                    int neg_r = r + radius/2;
                    int x_0_pos = x + (r + MIN_RADIUS) * cos(dir.at<float>(y,x)); // adding min radius to keep it between the valid range
                    int y_0_pos = y + (r + MIN_RADIUS) * sin(dir.at<float>(y,x));

                    int x_0_neg = x - (r + MIN_RADIUS) * cos(dir.at<float>(y,x));
                    int y_0_neg = y - (r + MIN_RADIUS) * sin(dir.at<float>(y,x));

                    // don't add circles whose centres are outside
                    if (x_0_pos >= 0 &&  x_0_pos < thr.cols && y_0_pos >= 0 && y_0_pos < thr.rows){
                        hough_space[x_0_pos][y_0_pos][r]++;
                    }

                    if (x_0_neg >= 0 &&  x_0_neg < thr.cols && y_0_neg >= 0 && y_0_neg < thr.rows){
                        hough_space[x_0_neg][y_0_neg][neg_r]++;
                    }
                }
            }
        }
    }

    Mat temp2D(thr.rows, thr.cols, CV_32FC1, Scalar(0));
    Mat hough2D(thr.rows, thr.cols, CV_8UC1, Scalar(0));

    for(int y = 0; y < centreY; y++){
        for(int x = 0; x < centreX; x++){
            for(int r = 0; r < radius; r++){
                temp2D.at<float>(y, x) += hough_space[x][y][r];
            }
        }
    }
    normalize(temp2D, hough2D, 0, 255, NORM_MINMAX);
    imwrite("hough_space.jpg", hough2D);
    // std::cout << hough2D.rows << '\n';
    // std::cout << hough2D.cols << '\n';
    Mat centres = thresholdHough(THRESHOLD_HOUGH);

    return centres;
}

void overlayHough(Mat &original, Mat &hough_centres) {
    Mat overlay(original.rows, original.cols, CV_32FC1, Scalar(0));
    overlay = original + hough_centres;
    normalize(overlay, overlay, 0, 255, NORM_MINMAX);
    imwrite("overlay.jpg",overlay);
}

int main(int argc, char const *argv[]) {
    Mat original = imread(argv[1], 0);
    Mat mag = imread("mag.jpg",0);
    Mat thr = threshold(mag, THRESHOLD);

    int centreX = thr.cols, centreY = thr.rows, radius = 2*(MAX_RADIUS - MIN_RADIUS);
    int ***hough_space;
    hough_space = malloc3dArray(centreX, centreY, radius);

    FileStorage fs("dir.yml", FileStorage::READ);
    Mat dir;
    fs["tempDir"] >> dir;

    Mat hough_centres = hough(thr, dir, hough_space, centreX, centreY, radius);

    overlayHough(original, hough_centres);
    return 0;
}
