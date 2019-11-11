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
#
using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );
void drawTruth(Mat frame,int values[][4],int length);

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	int values[][4] = {{329,79,52,60}};
	int length = sizeof(values)/sizeof(values[0]);

	drawTruth(frame,values,length);
	// 4. Save Result Image
	imwrite( "detected.jpg", frame );
	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;
	// std::vector<tuple<int,int,int,int>> ground_vals;
       // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
		printf("{%d,%d,%d,%d}\n",faces[i].x,faces[i].y,faces[i].width,faces[i].height );
	}
}
void drawTruth(Mat frame,int values[][4],int length){
	for(int i = 0; i < length; i++){
		int x = values[i][0];
		int y = values[i][1];
		int width = values[i][2];
		int height = values[i][3];
		rectangle(frame,Point(x,y) ,Point(x+width,y+height),Scalar(0, 0, 255), 2);
	}
}
