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

//Mandrill3 to Go to mandrillRGB HSV to RGB cvtColor( image, image, CV_HSV2BGR);
//Mandrill2 to MandrillRGB Inverted Image  bitwise_not(image, image);
Mat translateImg(Mat &img, int offsetx, int offsety){
    Mat trans_mat = (Mat_<double>(2,3) << 1, 0, offsetx, 0, 1, offsety);
    warpAffine(img,img,trans_mat,img.size());
    return img;
}

int main() {

  // Read image from file
  Mat image = imread("mandrill0.jpg", 1);
  Mat image2 = imread("mandrill1.jpg", 1);
  Mat redLayer = imread("mandrill1.jpg", 1);

  //shift red layer by offset
  uchar shift_val = 32;
  redLayer = translateImg(redLayer,shift_val,shift_val);

  //Code to recover mandrill0 (RGB Channels got mixed )
  for(int y=0; y<image.rows; y++) {
   for(int x=0; x<image.cols; x++) {
     uchar pixelblue = image.at<Vec3b>(y,x)[0]; //B
     uchar pixelgreen = image.at<Vec3b>(y,x)[1]; //G
     uchar pixelred = image.at<Vec3b>(y,x)[2]; //R
     image.at<Vec3b>(y,x)[0] = pixelred; //B -> R
     image.at<Vec3b>(y,x)[1] = pixelblue; //G -> B
     image.at<Vec3b>(y,x)[2] = pixelgreen; //R -> G
     //mandrill 1 recovery code red layer needs to be shifted
     uchar red_val = image2.at<Vec3b>(y,x)[2];
     //extract only red channel
     redLayer.at<Vec3b>(y,x)[0] = 0;
     redLayer.at<Vec3b>(y,x)[1] = 0;
     //turn off red channel for original image
     image2.at<Vec3b>(y,x)[2] = 0;
     //replace original red channel with shifted values
     image2.at<Vec3b>(y,x)[2] = redLayer.at<Vec3b>(y,x)[2];
} }
  //Save recovered image
  imwrite("recovered.jpg", image);
  imwrite("recovered1.jpg", image2);

  return 0;
}
