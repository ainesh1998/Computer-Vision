#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <math.h>

#include "Hough_Helper.h"

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
void HoughHelper::sobel(Mat &gray_image, Mat &mag_image, Mat &dir_image){
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

Mat HoughHelper::threshold(string input, string output, int threshVal) {
    Mat image = imread(input, 0);
    Mat thr(image.rows,image.cols,CV_8UC1,Scalar(0));

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
    imwrite(output,thr);
    return thr;
}

void HoughHelper::overlayHough(Mat &original, Mat &hough_centres, string name) {
    Mat overlay(original.rows, original.cols, CV_32FC1, Scalar(0));
    overlay = original + hough_centres;
    normalize(overlay, overlay, 0, 255, NORM_MINMAX);
    imwrite(name,overlay);
}
