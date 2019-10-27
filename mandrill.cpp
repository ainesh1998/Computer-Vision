/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - thr.cpp
// TOPIC: RGB explicit thresholding
//
// Getting-Started-File for OpenCV
// University of Bristol
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup


using namespace cv;
using namespace std;

//Mandrill3 t/R to Go M/R to GandrillRGB HSV to RGB cvtColor( image, image, CV_HSV2BGR);
//Mandrill2 to MandrillRGB Inverted Image  bitwise_not(image, image);

int main() {

  // Read image from file
  Mat image = imread("mandrill0.jpg", 1);
  //Code to recover mandrill0 (RGB Channels got mixed )
  for(int y=0; y<image.rows; y++) {
   for(int x=0; x<image.cols; x++) {
     uchar pixelblue = image.at<Vec3b>(y,x)[0]; //B
     uchar pixelgreen = image.at<Vec3b>(y,x)[1]; //G
     uchar pixelred = image.at<Vec3b>(y,x)[2]; //R
     image.at<Vec3b>(y,x)[0] = pixelred; //B -> R
     image.at<Vec3b>(y,x)[1] = pixelblue; //G -> B
     image.at<Vec3b>(y,x)[2] = pixelgreen; //R -> G
} }

  //Save recovered image
  imwrite("recovered.jpg", image);

  return 0;
}
