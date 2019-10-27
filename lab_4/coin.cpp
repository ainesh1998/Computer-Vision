// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;
using namespace std;

float convolution(Mat &img,int kernel[3][3],int y, int x){
    int val = 0;
    for(int a = -1; a <= 1; a++){
        for(int b = -1; b <= 1; b++){
            val += kernel[a + 1][b + 1] * img.at<uchar>(y - a,x - b);
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
    int grad_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int grad_y[3][3] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};


    Mat temp(image.rows,image.cols, CV_32FC1,Scalar(0));
    Mat resultX(image.rows,image.cols, CV_8UC1,Scalar(0));



    for(int y = 1; y < image.rows - 1; y++){
        for(int x = 1; x < image.cols - 1; x++){
            float new_point = convolution(gray_image,grad_x,y,x);
            temp.at<float>(y,x) = new_point;
            // if (new_point < 0) std::cout << new_point << '\n';

        }
    }
    normalize(temp,resultX,0,255,NORM_MINMAX);
    imwrite("grad_x.jpg",resultX);
    return 0;
}


/*

The problem is that the kernel has negative values this time - so putting these negative values into uchars mess them up. We should somehow try to
make a matrix with these negative values (not sure if 'Mat's take sign numbers), then normalise this

*/
