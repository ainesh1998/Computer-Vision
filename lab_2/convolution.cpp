#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup


using namespace cv;
using namespace std;
uchar convolution(Mat &img,int kernel[3][3],int y, int x){
    int val = 0;
    for(int a = -1; a <= 1; a++){
        for(int b = -1; b <= 1; b++){
            val += kernel[a + 1][b + 1] * img.at<uchar>(y - a,x - b);
        }
    }
    // if(val/9 > 255){
    //     return 255;
    // }
    return (uchar)(val/9);
}
int main(int argc, char const *argv[]) {
    Mat image = imread("mandrill.jpg",1);
    Mat gray_image;
    cvtColor(image,gray_image,CV_BGR2GRAY);
    // std::cout << gray_image.channels() << '\n';
    int kernel[3][3];
    Mat convul_image(image.rows - 2,image.cols- 2,CV_8UC1, Scalar(0));
    //Fill kernel with 1/9s
    for(int y = 0; y < 3; y++){
        for(int x= 0; x < 3; x++){
            kernel[y][x]= 1;
        }
    }

    for(int y = 1; y < image.rows - 1; y++){
        for (int x = 1;x < image.cols - 1;x++) {
            uchar new_point = convolution(gray_image,kernel,y,x);
            convul_image.at<uchar>(y,x) = new_point;
        }
    }
    // convul_image = conv2(image,kernel);
    // std::cout << image << '\n';
    imwrite("convolution.jpg",convul_image);
    return 0;
}
