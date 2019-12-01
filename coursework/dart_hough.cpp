/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include "Hough_Circle.h"
#include "Hough_Line.h"

#define IOU_THRESHOLD 0.3
#define RECT_CENTRE_THRESHOLD 0.38
#define CIRCLE_RECT_RATIO 0.09
#define INTERSECT_POINT_RADIUS 15

using namespace std;
using namespace cv;

/** Function Headers */
vector<Rect> detectAndDisplay( Mat frame );
//combine viola jones predictions with hough space predictions for stronger classifier
vector<Rect> violaHough(Mat centres, Mat line_intersections, vector<Rect> dartboards, Mat radii);
bool compareRect(Rect rect1, Rect rect2);
void drawTruth(Mat frame,int values[][4],int length);
double true_pos_rate(vector<Rect> predictions,int truth_values[][4],int truth_length);
double intersectionOverUnion(Rect prediction,int truth_value[4]);
double calc_f1_score(vector<Rect> predictions,int truth_values[][4],int truth_length,double tpr);
void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurredOutput);
Mat sharpen(Mat &image, int iterations);

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;

/** @function main */
int main( int argc, const char** argv )
{
	HoughCircle houghCircle;
	HoughLine houghLine;

    // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat sharpenedFrame = sharpen(frame,0);
	equalizeHist(sharpenedFrame,sharpenedFrame);

	Mat gray_image;
	cvtColor(frame, gray_image, CV_BGR2GRAY);

	Mat radii = houghCircle.circle_detect(gray_image);
	Mat centres = imread("circle_space_thr.jpg",0);
	Mat line_intersections = houghLine.line_detect(sharpenedFrame);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	vector<Rect> dartboards = detectAndDisplay( frame );

	vector<Rect> predictions = violaHough(centres,line_intersections,dartboards,radii);

	std::cout << predictions.size() << std::endl;
	for( int i = 0; i < predictions.size(); i++ )
	{
		for(int y = 0 ; y < centres.rows; y++){
			for(int x = 0; x < centres.cols; x++){
				if (centres.at<uchar>(y,x) == 255) {
					circle(frame,Point(x,y),1,Scalar(255,110,199),2);
				}
				if (line_intersections.at<uchar>(y,x) == 255) {
					circle(frame,Point(x,y),1,Scalar(255,0,0),2);
				}
			}
		}
		circle(frame, (predictions[i].tl()+predictions[i].br())*0.5, 0.5*(1-2*RECT_CENTRE_THRESHOLD)*predictions[i].height, Scalar(0,255,0),2);
		circle(frame, (predictions[i].tl()+predictions[i].br())*0.5, 10, Scalar(0,255,0),2);
		rectangle(frame, Point(predictions[i].x, predictions[i].y), Point(predictions[i].x + predictions[i].width, predictions[i].y + predictions[i].height), Scalar( 0, 255, 0 ), 2);
	}
	int ground_truth_vals[][4] = {{64,249,66,94},{840,215,121,126}};
	int length = sizeof(ground_truth_vals)/sizeof(ground_truth_vals[0]);
	drawTruth(frame,ground_truth_vals,length);
	double tpr = true_pos_rate(predictions,ground_truth_vals,length);
	printf("true pos rate = %f \n",tpr );
	double f1_score = calc_f1_score(predictions,ground_truth_vals,length,tpr);
	printf("f1 score = %f \n",f1_score);
	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

/** @function detectAndDisplay */
vector<Rect> detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	return faces;
}

vector<Rect> violaHough(Mat centres, Mat line_intersections, vector<Rect> dartboards, Mat radii){
	vector<Rect> predictions;
	vector<Rect> final;

	for(int i = 0; i < dartboards.size(); i++){
		int innerCountCircles = 0;
		int innerCountLines = 0;
		Point centre = (dartboards[i].tl() + dartboards[i].br()) * 0.5;
		int r = (1 - 2*RECT_CENTRE_THRESHOLD) * dartboards[i].height/2;
		int temp = RECT_CENTRE_THRESHOLD * dartboards[i].height;
		float centreArea = M_PI*r*r;
		float wholeArea = dartboards[i].height * dartboards[i].width;

		for(int y = dartboards[i].y +temp; y < dartboards[i].y + dartboards[i].height-r; y++){
			for(int x = dartboards[i].x + temp; x < dartboards[i].x + dartboards[i].width-r; x++){
				// check for the circle inside the rect centre
				int dist = (centre.x-x)*(centre.x-x) + (centre.y-y)*(centre.y-y);

				if(dist < r*r) {
					if(centres.at<uchar>(y,x) == 255) {
						innerCountCircles++;
					}
				}
			}
		}

		temp = dartboards[i].height/2 - INTERSECT_POINT_RADIUS;
		for(int y = dartboards[i].y +temp; y < dartboards[i].y + dartboards[i].height-r; y++){
			for(int x = dartboards[i].x + temp; x < dartboards[i].x + dartboards[i].width-r; x++){
				// check for the circle inside the rect centre
				int dist = (centre.x-x)*(centre.x-x) + (centre.y-y)*(centre.y-y);

				if(dist < r*r) {
					if(line_intersections.at<uchar>(y,x) == 255) {
						innerCountLines++;
					}
				}
			}
		}

		float wholeCountCircles = 0.0;
		float wholeCountLines = 0.0;

		for(int y = 0 ; y < centres.rows; y++){
			for(int x = 0; x < centres.cols; x++){
				if (centres.at<uchar>(y,x) == 255) {
					wholeCountCircles += 1;

				}
				if (line_intersections.at<uchar>(y,x) == 255) {
					wholeCountLines += 1;
				}
			}
		}
		float circleRatio = innerCountCircles/(wholeCountCircles);
		float lineRatio = innerCountLines/(wholeCountLines);
		float actualRatio = (circleRatio + lineRatio)/2;
		std::cout << actualRatio << '\n';
		if (actualRatio > CIRCLE_RECT_RATIO) predictions.push_back(dartboards[i]);
	}

	// CHECK FOR NESTED RECTANGLES
	vector<vector<Rect> > alreadySeenPairs; // a vector of tuples containing pairs already processed

	for (int i = 0; i < predictions.size(); i++) {
		bool noNesting = true; // true if predictions[i] is not anywhere in alreadySeenPairs

		for (int j = 0; j < predictions.size(); j++) {
			// check for nested rectangles

			// intersection of predictions[i] and predictions[j], equal to the smaller rectangle if nested
			int intersectionArea = (predictions[i] & predictions[j]).area();

			if (intersectionArea>0) {
				// Rect smallerOne = intersectionArea == predictions[i].area() ? predictions[i] : predictions[j];
				// Rect largerOne = intersectionArea == predictions[i].area() ? predictions[j] : predictions[i];

				// get the difference between each radii and the average radii at that centre
				int dist1 = abs(radii.at<float>(predictions[i].y + predictions[i].height/2, predictions[i].x + predictions[i].width/2) - predictions[i].width/2);
				int dist2 = abs(radii.at<float>(predictions[j].y + predictions[j].height/2, predictions[j].x + predictions[j].width/2) - predictions[j].width/2);

				// the radius that is closer to the actual one is the better rectangle
				Rect theOne = dist1 < dist2 ? predictions[i] : predictions[j];

				// true if (predictions[i],predictions[j]) has already been seen
				bool isSeen = false;

				for (int k = 0; k < alreadySeenPairs.size(); k++) {
					isSeen = isSeen || compareRect(alreadySeenPairs[k][0],predictions[i])&&compareRect(alreadySeenPairs[k][1],predictions[j]);
					noNesting = noNesting && !(compareRect(alreadySeenPairs[k][0],predictions[i])||compareRect(alreadySeenPairs[k][1],predictions[i]));
				}

				if (!isSeen && i != j) {
					// add the pair
					final.push_back(theOne);
					vector<Rect> pair1;
					vector<Rect> pair2;
					pair1.push_back(predictions[i]);
					pair1.push_back(predictions[j]);
					pair2.push_back(predictions[j]);
					pair2.push_back(predictions[i]);
					alreadySeenPairs.push_back(pair1);
					alreadySeenPairs.push_back(pair2);
					noNesting = false; // it is nested so we cannot add it at the end
				}
			}
		}
		if (noNesting) final.push_back(predictions[i]);
	}
	return final;
}

// Checks if two Rects are equal
bool compareRect(Rect rect1, Rect rect2) {
	return rect1.x == rect2.x && rect1.y == rect2.y && rect1.height == rect2.height && rect1.width == rect2.width;
}

///// FROM face.cpp /////

void drawTruth(Mat frame,int values[][4],int length){
	for(int i = 0; i < length; i++){
		int x = values[i][0];
		int y = values[i][1];
		int width = values[i][2];
		int height = values[i][3];
		rectangle(frame,Point(x,y) ,Point(x+width,y+height),Scalar(0, 0, 255), 2);
	}
}
//given predictions and truth values calculate the true positive rate (no of correct faces/no of valid faces)
double true_pos_rate(vector<Rect> predictions,int truth_values[][4],int truth_length){
	int detected;
	double iou_scores[predictions.size() * truth_length];
	for(int i = 0; i < predictions.size();i++){
		for(int j = 0; j < truth_length; j++){
			//compare each prediction with every truth value
			//if they don't overlap IOU = 0
			double iou = intersectionOverUnion(predictions[i],truth_values[j]);
			iou_scores[truth_length * i + j] = iou;
		}
	}
	//calculate no of detected faces
	for(int i = 0; i < predictions.size() * truth_length; i++){
		if(iou_scores[i] > IOU_THRESHOLD) detected++;
	}
	//detected = true positives
	return (double)detected/(double)truth_length;
}
/*
Given predictions and truth values, calculates the f1 score
f1 score = 2 x ((precision * recall)/(precision + recall))
precision = true positives/(true positives + false positives)
recall (tpr) = true positives/(true positives + false negative)
true positives + false negatives = truth_values
therefore recall = tpr
*/
double calc_f1_score(vector<Rect> predictions,int truth_values[][4],int truth_length,double tpr){
	double recall = tpr;
	int true_positives = recall * truth_length;
	double precision = (double)true_positives/(double)predictions.size();
	printf("precision = %f\n",precision );
	double f1_score = 2 * (precision * recall)/(precision + recall);
	return f1_score;
}
//Compute the intersection over union between prediction and truth value boxes
double intersectionOverUnion(Rect prediction,int truth_value[4]){
	//determine coordinates of intersection rectangle
	int x0 = max(prediction.x,truth_value[0]);
	int y0 = max(prediction.y,truth_value[1]);
	int x1 = min(prediction.x + prediction.width,truth_value[0] + truth_value[2]);
	int y1 = min(prediction.y+ prediction.height,truth_value[1]+ truth_value[3]);
	//find area of intersection rectangle
	//0 is in case there's no intersect
	double intersect_area = max(0,(x1 - x0)) * max(0,(y1 - y0));
	double pred_area = prediction.width * prediction.height;
	double truth_area = truth_value[2] * truth_value[3];
	double iou = intersect_area/(pred_area + truth_area - intersect_area);
	// printf("%f\n",iou );
	return iou;
}

/// FROM LAB 3 ///

void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurredOutput)
{
	// intialise the output using the input
	blurredOutput.create(input.size(), input.type());

	// create the Gaussian kernel in 1D
	cv::Mat kX = cv::getGaussianKernel(size, -1);
	cv::Mat kY = cv::getGaussianKernel(size, -1);

	// make it 2D multiply one by the transpose of the other
	cv::Mat kernel = kX * kY.t();

	//CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
	//TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

       // SET KERNEL VALUES
	for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) {
	  for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
           kernel.at<double>(m+ kernelRadiusX, n+ kernelRadiusY) = (double) 1.0/(size*size);

       }

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convoltion
	for ( int i = 0; i < input.rows; i++ )
	{
		for( int j = 0; j < input.cols; j++ )
		{
			double sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernalval = kernel.at<double>( kernelx, kernely );

					// do the multiplication
					sum += imageval * kernalval;
				}
			}
			// set the output value as the sum of the convolution
			blurredOutput.at<uchar>(i, j) = (uchar) sum;
		}
	}
}

// unsharp masking
Mat sharpen(Mat &frame, int iterations) {
	Mat gray_image;
	Mat carBlurred(frame.rows,frame.cols,CV_32FC1,Scalar(0));
	Mat sharpened(frame.rows,frame.cols,CV_32FC1,Scalar(0));
	Mat final(frame.rows,frame.cols,CV_8UC1,Scalar(0));

	cvtColor(frame, gray_image, CV_BGR2GRAY );
	GaussianBlur(gray_image,5,carBlurred);

	sharpened = gray_image;

	for (int i = 0; i < iterations; i++) {
		sharpened += gray_image - carBlurred;
	}

	normalize(sharpened,final,0,255,NORM_MINMAX);
	imwrite("sharpened.jpg",final);
	return final;
}
