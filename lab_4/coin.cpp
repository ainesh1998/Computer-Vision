// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;
using namespace std;


float convolution(Mat &img,Mat &kernel,int y, int x){
    int val = 0;
    for(int a = -1; a <= 1; a++){
        for(int b = -1; b <= 1; b++){
            val += kernel.at<int>(a + 1,b + 1) * img.at<uchar>(y - a,x - b);
        }
    }
    return val;
}


void sobel(Mat &image){

}

int main(int argc, char const *argv[]) {
    const char* image_name = argv[1];
    Mat image = imread(image_name,1);
    Mat gray_image;

    cvtColor(image,gray_image,CV_BGR2GRAY);
    // int grad_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    Mat gradX = (Mat_<int>(3,3) << -1,0,1,-2,0,2,-1,0,1);
    Mat gradY = gradX.t();

    Mat tempX(image.rows,image.cols, CV_32FC1,Scalar(0));
    Mat tempY(image.rows,image.cols, CV_32FC1,Scalar(0));
    Mat tempMag(image.rows,image.cols, CV_32FC1,Scalar(0));
    Mat tempDir(image.rows,image.cols, CV_32FC1,Scalar(0));

    Mat resultX(image.rows,image.cols, CV_8UC1,Scalar(0));
    Mat resultY(image.rows,image.cols, CV_8UC1,Scalar(0));
    Mat resultMag(image.rows,image.cols, CV_8UC1,Scalar(0));
    Mat resultDir(image.rows,image.cols, CV_8UC1,Scalar(0));

    for(int y = 1; y < image.rows - 1; y++){
        for(int x = 1; x < image.cols - 1; x++){
            float dx = convolution(gray_image,gradX,y,x);
            float dy = convolution(gray_image,gradY,y,x);
            tempX.at<float>(y,x) = dx;
            tempY.at<float>(y,x) = dy;
            tempMag.at<float>(y,x) = sqrt(dx * dx + dy * dy);
            tempDir.at<float>(y,x) = atan(dy/dx);
        }
    }
    normalize(tempX,resultX,0,255,NORM_MINMAX);
    normalize(tempY,resultY,0,255,NORM_MINMAX);
    normalize(tempMag,resultMag,0,255,NORM_MINMAX);
    normalize(tempDir,resultDir,0,255,NORM_MINMAX);
    imwrite("grad_x.jpg",resultX);
    imwrite("grad_y.jpg",resultY);
    imwrite("mag.jpg",resultMag);
    imwrite("dir.jpg",resultDir);
    return 0;
}


/*

The problem is that the kernel has negative values this time - so putting these negative values into uchars mess them up. We should somehow try to
make a matrix with these negative values (not sure if 'Mat's take sign numbers), then normalise this

*/
