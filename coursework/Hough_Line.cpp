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
#include "Hough_Helper.h"
#define THRESHOLD_LINES 80 // thresholding the original magnitude image for hough lines
#define THRESHOLD_LINE_SPACE 35 // thresholding the hough space for hough lines
#define THRESHOLD_INTERSECTIONS 100
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
    imwrite("line_space.jpg", displayHough);
    HoughHelper h;
    Mat thresholded = h.threshold("line_space.jpg","line_space_thr.jpg",THRESHOLD_LINE_SPACE);
    return thresholded;
}

float eval(int rho,float theta,int x){
    return (rho - x * cos(theta))/sin(theta);
}

Mat drawLines(Mat hough_lines, Mat image) {
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
    imwrite("intersection_space.jpg",line_space);
    HoughHelper h;
    Mat thresholded = h.threshold("intersection_space.jpg","intersection_thr.jpg",THRESHOLD_INTERSECTIONS);
    return thresholded;
}

Mat HoughLine::line_detect(Mat &image1) {
    Mat image;
    cvtColor(image1,image,CV_BGR2GRAY);
    HoughHelper h;

    //get magnitude image and direction image
    Mat dir_image = h.sobel(image);

    //threshold magnitude image for hough transform
    Mat thr;
    thr = h.threshold("mag.jpg","thr_line.jpg",THRESHOLD_LINES);

    //perform hough transform
    int rho = sqrt(image.rows*image.rows + image.cols*image.cols);
    int theta = MAX_THETA - MIN_THETA;
    int **hough_space;
    hough_space = malloc2dArray(rho,theta);

    // Build the hough space
    Mat hough_lines = hough_builder_lines(thr,dir_image,hough_space,rho,theta);
    Mat intersections = drawLines(hough_lines,image);
    return intersections;
}
