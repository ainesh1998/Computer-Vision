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

#define IOU_THRESHOLD 0.4
#define RECT_CENTRE_THRESHOLD 0.4

using namespace std;
using namespace cv;

/** Function Headers */
vector<Rect> detectAndDisplay( Mat frame );
//combine viola jones predictions with hough space predictions for stronger classifier
vector<Rect> violaHough(Mat centres, Mat line_intersections, vector<Rect> dartboards, Mat radii);
void drawTruth(Mat frame,int values[][4],int length);
double true_pos_rate(vector<Rect> predictions,int truth_values[][4],int truth_length);
double intersectionOverUnion(Rect prediction,int truth_value[4]);
double calc_f1_score(vector<Rect> predictions,int truth_values[][4],int truth_length,double tpr);

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

	Mat radii = houghCircle.circle_detect(frame);
	Mat centres = imread("circle_space_thr.jpg",0);
	Mat line_intersections = houghLine.line_detect(frame);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	vector<Rect> dartboards;
	dartboards = detectAndDisplay( frame );

	vector<Rect> predictions = violaHough(centres,line_intersections,dartboards,radii);

	std::cout << predictions.size() << std::endl;
	for( int i = 0; i < predictions.size(); i++ )
	{
		rectangle(frame, Point(predictions[i].x, predictions[i].y), Point(predictions[i].x + predictions[i].width, predictions[i].y + predictions[i].height), Scalar( 0, 255, 0 ), 2);
	}
	int ground_truth_vals[][4] = {{250,166,152,154}};
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
		bool found = false;
		for(int y = dartboards[i].y +RECT_CENTRE_THRESHOLD * dartboards[i].height; y < dartboards[i].y + (1-RECT_CENTRE_THRESHOLD)* dartboards[i].height; y++){
			for(int x = dartboards[i].x + RECT_CENTRE_THRESHOLD * dartboards[i].width; x < dartboards[i].x + (1-RECT_CENTRE_THRESHOLD)*dartboards[i].width; x++){
				if(!found){
					if(centres.at<uchar>(y,x) == 255 && line_intersections.at<uchar>(y,x) == 255) {
						predictions.push_back(dartboards[i]);
						found = true;
					}
				}
			}
		}
	}

	std::cout << predictions.size() << '\n';
	vector<Rect> chucked;

	// check for rectangles inside another rectangle, store already processed ones as pairs in a vector 

	for (int i = 0; i < predictions.size(); i++) {
		bool isOverlap = false;

		for (int j = 0; j < predictions.size(); j++) {
			Point centre1 = (predictions[i].tl() + predictions[i].br()) * 0.5;
			Point centre2 = (predictions[j].tl() + predictions[j].br()) * 0.5;
			float total = 1/(radii.at<float>(centre1.y,centre1.x) + radii.at<float>(centre2.y,centre2.x));
			Point centre = (radii.at<float>(centre1.y,centre1.x)*centre1 + radii.at<float>(centre2.y,centre2.x)*centre2) * total;
			int dist1 = abs(radii.at<float>(centre.y, centre.x) - predictions[i].width/2);
			int dist2 = abs(radii.at<float>(centre.y, centre.x) - predictions[j].width/2);
			bool choosingRect = dist1 < dist2;
			Rect smallerOne = choosingRect ? predictions[i] : predictions[j];
			Rect largerOne = !choosingRect ? predictions[i] : predictions[j];

			if (i != j && find(final.begin(), final.end(), predictions[i]) == final.end() && find(final.begin(), final.end(), predictions[j]) == final.end()) {
				if ((abs(centre1.x - centre2.x) < 0.1*predictions[i].width && abs(centre1.y - centre2.y)) < 0.1*predictions[i].height) {

					final.push_back(smallerOne);
					chucked.push_back(largerOne);
					isOverlap = true;
				}
			}
			else {
				final.push_back(predictions[j]);
			}
		}
		if (!isOverlap && find(chucked.begin(), chucked.end(), predictions[i]) == chucked.end())  {
			final.push_back(predictions[i]);
		}
	}
	groupRectangles(final,1,0);
	// final = predictions;
	return final;
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
