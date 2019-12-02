#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <math.h>

#include "Hough_Circle.h"
#include "Hough_Helper.h"
#define THRESHOLD_CIRCLES 80 // thresholding the original magnitude image for hough circles
#define THRESHOLD_HOUGH_CENTRES 120 // thresholding the hough space for hough circles
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
    Mat radii(thr.rows, thr.cols, CV_32FC1, Scalar(0));

    for(int y = 0; y < centreY; y++){
        for(int x = 0; x < centreX; x++){
            int temp = 0;
            int total = 0;
            for(int r = 0; r < radius; r++){
                hough2D.at<float>(y, x) += hough_space[x][y][r];
                temp += (r + MIN_RADIUS) * hough_space[x][y][r];
                total += hough_space[x][y][r];
            }
            radii.at<float>(y,x) = total > 0 ? temp/total : 0;
        }
    }
    normalize(hough2D, displayHough, 0, 255, NORM_MINMAX);
    imwrite("circle_space.jpg", displayHough);
    HoughHelper h;
    Mat thresholded = h.threshold("circle_space.jpg","circle_space_thr.jpg",THRESHOLD_HOUGH_CENTRES);
    imwrite("radii.jpg",radii);
    return radii;
}

Mat HoughCircle::circle_detect(Mat &image) {
    //get magnitude image and direction image
    HoughHelper h;
    Mat dir_image = h.sobel(image);

    //threshold magnitude image for hough transform
    Mat thr;
    thr = h.threshold("mag.jpg","thr_circle.jpg",THRESHOLD_CIRCLES);

    //perform hough transform
    int centreX = thr.cols, centreY = thr.rows, radius = (MAX_RADIUS - MIN_RADIUS);
    int ***hough_space;
    hough_space = malloc3dArray(centreX, centreY, radius);

    // Build the hough space
    Mat radii = hough_builder_circles(thr,dir_image,hough_space,centreX,centreY,radius);
    Mat hough_centres = imread("circle_space_thr.jpg",0);
    h.overlayHough(image, hough_centres, "circles_detected.jpg");
    return radii;
}
