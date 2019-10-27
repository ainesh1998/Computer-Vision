#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int grid_size = 3;

// A utility function to swap two elements
void swap(uchar* a, uchar* b)
{
	uchar t = *a;
	*a = *b;
	*b = t;
}

/* This function takes last element as pivot, places
the pivot element at its correct position in sorted
array, and places all smaller (smaller than pivot)
to left of pivot and all greater elements to right
of pivot */
int partition (uchar arr[], int low, int high)
{
	uchar pivot = arr[high]; // pivot
	int i = (low - 1); // Index of smaller element

	for (int j = low; j <= high - 1; j++)
	{
		// If current element is smaller than the pivot
		if (arr[j] < pivot)
		{
			i++; // increment index of smaller element
			swap(&arr[i], &arr[j]);
		}
	}
	swap(&arr[i + 1], &arr[high]);
	return (i + 1);
}

/* The main function that implements QuickSort
arr[] --> Array to be sorted,
low --> Starting index,
high --> Ending index */
void quickSort(uchar arr[], int low, int high)
{
	if (low < high)
	{
		/* pi is partitioning index, arr[p] is now
		at right place */
		int pi = partition(arr, low, high);

		// Separately sort elements before
		// partition and after partition
		quickSort(arr, low, pi - 1);
		quickSort(arr, pi + 1, high);
	}
}

uchar median(uchar numbers[25]){
    // sort(numbers,numbers + 9);
    quickSort(numbers,0,24);

    return numbers[12];
}

void getNeighbours(uchar grid[25], Mat& image, int y, int x){
    for (int a = -2; a <= 2; a++) {
        for (int b = -2; b <= 2; b++) {
            int index = 5*(a+2)+(b+2);
            if (y+a < 0 || y+a > image.rows-1 || x+b < 0 || x+b > image.cols-1) {
                grid[index] = 0;
            }
            else {
                grid[index] = image.at<uchar>(y+a, x+b);
            }
        }
    }
}

int main(int argc, char const * argv[]) {
    // LOADING THE IMAGE
    const char* imageName = argv[1];

    Mat image;
    image = imread( imageName, 1 );
    //Median_Filtering
    Mat gray_image;

    cvtColor( image, gray_image, CV_BGR2GRAY );

    Mat new_image(gray_image.rows, gray_image.cols, CV_8UC1,Scalar(0));

    for(int y = 0; y < gray_image.rows; y++){
        for(int x = 0; x < gray_image.cols; x++){
            uchar grid[25];
            getNeighbours(grid, gray_image, y, x);
            new_image.at<uchar>(y, x) = median(grid);
        }
    }

    imwrite("med_filter.jpg", new_image);
    return 0;
}
