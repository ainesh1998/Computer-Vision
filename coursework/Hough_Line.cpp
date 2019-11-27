#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <math.h>

#include "Hough_Line.h"
#define THRESHOLD_LINES 80 // thresholding the original magnitude image for hough lines
#define THRESHOLD_LINE_SPACE 35 // thresholding the hough space for hough lines
#define MIN_THETA -180 // minimum angle represented in the hough line space
#define MAX_THETA 180 // maximum angle represented in the hough line space
#define DELTA_THETA 5 // range we consider for angles for each hough line


int **malloc2dArray(int rho, int theta){
    int i, j;
    int **array = (int **) malloc(rho * sizeof(int *));

    for (i = 0; i < rho; i++) {
        array[i] = (int *) malloc(theta * sizeof(int ));

    }
    return array;
}

float convolution2(Mat &img,Mat &kernel,int y, int x){
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

void sobel2(Mat &gray_image, Mat &mag_image, Mat &dir_image){
    Mat gradX = (Mat_<int>(3,3) << -1,0,1,-2,0,2,-1,0,1);
    Mat gradY = gradX.t();
    Mat gradX_image(gray_image.rows,gray_image.cols, CV_32FC1,Scalar(0));
    Mat gradY_image(gray_image.rows,gray_image.cols, CV_32FC1,Scalar(0));
    for(int y = 1; y < gray_image.rows - 1; y++){
        for(int x = 1; x < gray_image.cols - 1; x++){
            float dx = convolution2(gray_image,gradX,y,x);
            float dy = convolution2(gray_image,gradY,y,x);
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
    imwrite("mag1.jpg",displayMag);
    imwrite("dir1.jpg", displayDir);
}
Mat threshold2(Mat &image, int threshVal) {
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
Mat thresholdHough2(string img, int threshVal) {
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

Mat hough_builder_lines(Mat &thr, Mat &dir, int **hough_space, int rho, int theta) {
    //zero hough space
    // dir = image of direction of the gradient
    for(int i = 0; i < rho; i++){
        for(int j = 0; j < theta; j++){
            hough_space[i][j] = 0;
        }
    }
    for(int y = 0; y < thr.rows; y++){
        for(int x = 0; x < thr.cols; x++){
            if(thr.at<uchar>(y,x) == 255){
                //hough voting
                float actual_grad = dir.at<float>(y,x);
                int degreeGradIndex = (actual_grad*180)/M_PI - MIN_THETA;

                for (int i = degreeGradIndex-DELTA_THETA; i <= degreeGradIndex+DELTA_THETA; i++) {
                    int corrected_degree = (i + theta)%theta;
                    float corrected_rad = (corrected_degree+MIN_THETA)*M_PI/180;
                    int rho_val = x * cos(corrected_rad) + y * sin(corrected_rad);

                    if (rho_val >= 0 && rho_val < rho) {
                        hough_space[rho_val][corrected_degree]++;
                    }
                }
            }
        }
    }

    // Create 2D Hough space image
    Mat hough2D(rho, theta, CV_32FC1, Scalar(0));
    Mat displayHough(rho, theta, CV_8UC1, Scalar(0));
    for (int i = 0; i < rho; i++) {
        for (int j = 0; j < theta; j++) {
            hough2D.at<float>(i,j) = hough_space[i][j];
        }
    }
    normalize(hough2D, displayHough, 0, 255, NORM_MINMAX);
    imwrite("hough_space_lines.jpg", displayHough);
    Mat thresholded = thresholdHough2("hough_space_lines.jpg",THRESHOLD_LINE_SPACE);
    imwrite("lines_thresholded.jpg",thresholded);
    return thresholded;
}

void overlayHough2(Mat &original, Mat &hough_centres) {
    Mat overlay(original.rows, original.cols, CV_32FC1, Scalar(0));
    overlay = original + hough_centres;
    normalize(overlay, overlay, 0, 255, NORM_MINMAX);
    imwrite("hough_detected.jpg",overlay);
}

float eval(int rho,float theta,int x){
    return (rho - x * cos(theta))/sin(theta);
}

void drawLines(Mat hough_lines, Mat image) {
    Mat temp(image.rows,image.cols,CV_32FC1,Scalar(0));
    Mat line_space(image.rows,image.cols,CV_8UC1,Scalar(0));
    //hough_lines.rows = rho
    //hough_lines.cols = theta
    for(int y = 0; y < hough_lines.rows; y++){
        for(int x = 0; x < hough_lines.cols; x++){
            if(hough_lines.at<uchar>(y,x) == 255){
                float grad_rad = ((x+MIN_THETA)*M_PI)/180;

                for (int i = 0; i < image.cols; i++) {
                    int j = eval(y,grad_rad,i);
                    if (j >= 0 && j < image.rows) temp.at<float>(j,i)++;
                }
            }
        }
    }
    normalize(temp, line_space, 0, 255, NORM_MINMAX);
    imwrite("line_space.jpg",line_space);
}

void HoughLine::line_detect(Mat &image1) {
    Mat image;
    cvtColor(image1,image,CV_BGR2GRAY);

    //get magnitude image and direction image
    Mat mag_image(image.rows,image.cols, CV_32FC1);
    Mat dir_image(image.rows, image.cols, CV_32FC1);
    sobel2(image,mag_image,dir_image);

    //threshold magnitude image for hough transform
    Mat thr;
    Mat test_mag = imread("mag1.jpg",0);
    thr = threshold2(test_mag,THRESHOLD_LINES);

    //perform hough transform
    int rho = sqrt(image.rows*image.rows + image.cols*image.cols);
    int theta = MAX_THETA - MIN_THETA;
    int **hough_space;
    hough_space = malloc2dArray(rho,theta);

    // Build the hough space
    Mat hough_lines = hough_builder_lines(thr,dir_image,hough_space,rho,theta);
    drawLines(hough_lines,image);
}
