#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <math.h>

#include "Hough_Circle.h"
#define THRESHOLD_CIRCLES 80 // thresholding the original magnitude image for hough circles
#define THRESHOLD_HOUGH_CENTRES 80 // thresholding the hough space for hough circles
#define MAX_RADIUS 150 // maximum radius for circles in the hough circle space
#define MIN_RADIUS 15 // minimum radius for circles in the hough circle space

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

float convolution(Mat &img,Mat &kernel,int y, int x){
    int val = 0;
    for(int a = -1; a <= 1; a++){
        for(int b = -1; b <= 1; b++){
            val += kernel.at<int>(a + 1,b + 1) * img.at<uchar>(y - a,x - b);
        }
    }
    return val;
}

/*computes the mag and direction for image and stores in
  mag_image and dir_image respectively
*/
void sobel(Mat &gray_image, Mat &mag_image, Mat &dir_image){
    Mat gradX = (Mat_<int>(3,3) << -1,0,1,-2,0,2,-1,0,1);
    Mat gradY = gradX.t();
    Mat gradX_image(gray_image.rows,gray_image.cols, CV_32FC1,Scalar(0));
    Mat gradY_image(gray_image.rows,gray_image.cols, CV_32FC1,Scalar(0));
    for(int y = 1; y < gray_image.rows - 1; y++){
        for(int x = 1; x < gray_image.cols - 1; x++){
            float dx = convolution(gray_image,gradX,y,x);
            float dy = convolution(gray_image,gradY,y,x);
            gradX_image.at<float>(y,x) = dx;
            gradY_image.at<float>(y,x) = dy;
            mag_image.at<float>(y,x) = sqrt(dx * dx + dy * dy);
            dir_image.at<float>(y,x) = atan2(dy, dx);
        }
    }
    Mat displayMag(gray_image.rows,gray_image.cols, CV_8UC1,Scalar(0));
    Mat displayDir(gray_image.rows,gray_image.cols, CV_8UC1,Scalar(0));
    normalize(mag_image,displayMag,0,255,NORM_MINMAX);
    normalize(dir_image,displayDir,0,255,NORM_MINMAX);
    imwrite("mag.jpg",displayMag);
    imwrite("dir.jpg", displayDir);
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

Mat thresholdHough(string img, int threshVal) {
    Mat hough2D = imread(img, 0);
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
    return thr;
}

Mat hough_builder_circles(Mat &thr, Mat &dir, int ***hough_space, int centreX, int centreY, int radius) {
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
                for(int r = 0; r < radius; r++) {
                    int x_0_pos = x + (r + MIN_RADIUS) * cos(dir.at<float>(y,x)); // adding min radius to keep it between the valid range
                    int y_0_pos = y + (r + MIN_RADIUS) * sin(dir.at<float>(y,x));

                    int x_0_neg = x - (r + MIN_RADIUS) * cos(dir.at<float>(y,x));
                    int y_0_neg = y - (r + MIN_RADIUS) * sin(dir.at<float>(y,x));

                    // don't add circles whose centres are outside
                    if (x_0_pos >= 0 &&  x_0_pos < thr.cols && y_0_pos >= 0 && y_0_pos < thr.rows){
                        hough_space[x_0_pos][y_0_pos][r]++;
                    }

                    if (x_0_neg >= 0 &&  x_0_neg < thr.cols && y_0_neg >= 0 && y_0_neg < thr.rows){
                        hough_space[x_0_neg][y_0_neg][r]++;
                    }
                }
            }
        }
    }

    // Create 2D Hough space image
    Mat hough2D(thr.rows, thr.cols, CV_32FC1, Scalar(0));
    Mat displayHough(thr.rows, thr.cols, CV_8UC1, Scalar(0));
    for(int y = 0; y < centreY; y++){
        for(int x = 0; x < centreX; x++){
            for(int r = 0; r < radius; r++){
                hough2D.at<float>(y, x) += hough_space[x][y][r];
            }
        }
    }
    normalize(hough2D, displayHough, 0, 255, NORM_MINMAX);
    imwrite("hough_space.jpg", displayHough);
    Mat thresholded = thresholdHough("hough_space.jpg", THRESHOLD_HOUGH_CENTRES);
    imwrite("hough_centres.jpg",thresholded);
    return thresholded;
}

void overlayHough(Mat &original, Mat &hough_centres) {
    Mat overlay(original.rows, original.cols, CV_32FC1, Scalar(0));
    overlay = original + hough_centres;
    normalize(overlay, overlay, 0, 255, NORM_MINMAX);
    imwrite("hough_detected.jpg",overlay);
}


Mat HoughCircle::circle_detect(Mat &image1) {
    Mat image;
    cvtColor(image1,image,CV_BGR2GRAY);
    //get magnitude image and direction image
    Mat mag_image(image.rows,image.cols, CV_32FC1);
    Mat dir_image(image.rows, image.cols, CV_32FC1);
    sobel(image,mag_image,dir_image);

    //threshold magnitude image for hough transform
    Mat thr;
    Mat test_mag = imread("mag.jpg",0);
    thr = threshold(test_mag,THRESHOLD_CIRCLES);

    //perform hough transform
    int centreX = thr.cols, centreY = thr.rows, radius = (MAX_RADIUS - MIN_RADIUS);
    int ***hough_space;
    hough_space = malloc3dArray(centreX, centreY, radius);

    // Build the hough space
    Mat hough_centres = hough_builder_circles(thr,dir_image,hough_space,centreX,centreY,radius);
    overlayHough(image, hough_centres);
    return hough_centres;
}
