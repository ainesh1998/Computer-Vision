#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <math.h>

#include "Hough.h"
#define THRESHOLD_CIRCLES 80
#define THRESHOLD_LINES 35
#define THRESHOLD_HOUGH_CENTRES 80

#define MAX_RADIUS 150
#define MIN_RADIUS 15
#define MIN_THETA -180
#define MAX_THETA 180


// #define THRESHOLD_HOUGH_VOTES 5
// #define GROUP_THRESHOLD 2
// #define GROUP_EPS 0.5
int **malloc2dArray(int rho, int theta){
    int i, j;
    int **array = (int **) malloc(rho * sizeof(int *));

    for (i = 0; i < rho; i++) {
        array[i] = (int *) malloc(theta * sizeof(int ));

    }
    return array;
}

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

// Major axis of the ellipse is the horizontal axis
int ****malloc4dArray(int centreX, int centreY, int major_radius, int minor_radius){
    int i, j, k, l;
    int ****array = (int ****) malloc(centreX * sizeof(int ***));

    for (i = 0; i < centreX; i++) {
        array[i] = (int ***) malloc(centreY * sizeof(int **));
        for (j = 0; j < centreY; j++) {
            array[i][j] = (int **) malloc(major_radius * sizeof(int*));
            for (k = 0; k < major_radius; k++) {
                array[i][j][k] = (int *) malloc(minor_radius * sizeof(int));
            }
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
void sobel2(Mat &gray_image, Mat &mag_image, Mat &dir_image){
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
    imwrite("mag1.jpg",displayMag);
    imwrite("dir1.jpg", displayDir);
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
    imwrite("output.jpg",thr);
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
    return thresholdHough("hough_space.jpg", THRESHOLD_HOUGH_CENTRES);
}

Mat hough_builder_lines(Mat &thr, Mat &dir, int **hough_space, int rho, int theta) {
    //zero hough space
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
                int rho_val = x * cos(actual_grad) + y * sin(actual_grad);

                if (rho_val >= 0 && rho_val < rho) {
                    hough_space[rho_val][degreeGradIndex]++;
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
    return thresholdHough("hough_space_lines.jpg",THRESHOLD_LINES);
}

void overlayHough(Mat &original, Mat &hough_centres) {
    Mat overlay(original.rows, original.cols, CV_32FC1, Scalar(0));
    overlay = original + hough_centres;
    normalize(overlay, overlay, 0, 255, NORM_MINMAX);
    imwrite("hough_detected.jpg",overlay);
}

// Method 1 - take the hough space and get outer and inner circles with votes
// greater than the threshold, then group rectangles
vector<Rect> detectDartboards1(int ***hough_space, int centreX, int centreY, int radius) {
    // vector<Rect> dartboards;
    // vector<int> voteCount;
    // bool done = false;
    // bool temp = false;
    //
    // for(int i = 0; i < centreX; i++){
    //     for(int j = 0; j < centreY; j++){
    //         for(int k = 0; k < radius; k++){
    //             int actualRadius = (k + MIN_RADIUS);
    //
    //             if (hough_space[i][j][k] > THRESHOLD_HOUGH_VOTES) {
    //                 for (float a = 0.62; a <= 0.63; a += 0.01) {
    //                     int innerRadiusIndex = (int) actualRadius*a - MIN_RADIUS; // The index of the radius of the smaller circle inside the dartboard
    //                     // std::cout << innerRadiusIndex << '\n';
    //
    //                     // The circle is considered if it has enough votes, but also if the circle with the same centre but half the radius
    //                     // has enough votes
    //                     if (hough_space[i][j][innerRadiusIndex] > THRESHOLD_HOUGH_VOTES && k > MIN_RADIUS/a) {
    //                         // Create the rectangle
    //                         int x = i - actualRadius;
    //                         int y = j - actualRadius;
    //                         int width = actualRadius * 2;
    //                         Rect rectangle = Rect(x, y, width, width);
    //
    //                         dartboards.push_back(rectangle);
    //                         voteCount.push_back(hough_space[i][j][k]);
    //                         if (!done) {
    //                             std::cout << "test" << '\n';
    //                             temp = true;
    //                         }
    //                         break;
    //                     }
    //
    //                 }
    //             }
    //             if (temp) done = true;
    //         }
    //     }
    // }
    //
    // // Remove overlapping rectangles
    // groupRectangles(dartboards, voteCount, GROUP_THRESHOLD, GROUP_EPS);
    // return dartboards;
}

Mat Hough::circle_detect(Mat &image1) {
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
float eval(int rho,float theta,int x){
    return (rho - x * cos(theta))/sin(theta);
}
void drawLines(Mat hough_lines, Mat image) {
    Mat line_space(image.rows,image.cols,CV_8UC1,Scalar(0));
    //hough_lines.rows = rho
    //hough_lines.cols = theta
    for(int y = 0; y < hough_lines.rows; y++){
        for(int x = 0; x < hough_lines.cols; x++){
            if(hough_lines.at<uchar>(y,x) == 255){
                float grad_rad = ((x+MIN_THETA)*M_PI)/180;
                int y_0 = eval(y,grad_rad,0);
                int y_1 = eval(y,grad_rad,image.cols-1);
                line(line_space,Point(0,y_0),Point(image.cols-1,y_1),Scalar(255));
            }
        }
    }
    imwrite("line_space.jpg",line_space);
}

void Hough::line_detect(Mat &image1) {
    Mat image;
    cvtColor(image1,image,CV_BGR2GRAY);

    //get magnitude image and direction image
    Mat mag_image(image.rows,image.cols, CV_32FC1);
    Mat dir_image(image.rows, image.cols, CV_32FC1);
    sobel2(image,mag_image,dir_image);

    //threshold magnitude image for hough transform
    Mat thr;
    Mat test_mag = imread("mag1.jpg",0);
    thr = threshold(test_mag,THRESHOLD_CIRCLES);

    //perform hough transform
    int rho = sqrt(image.rows*image.rows + image.cols*image.cols);
    int theta = MAX_THETA - MIN_THETA;
    int **hough_space;
    hough_space = malloc2dArray(rho,theta);

    // Build the hough space
    Mat hough_lines = hough_builder_lines(thr,dir_image,hough_space,rho,theta);
    drawLines(hough_lines,image);
}
