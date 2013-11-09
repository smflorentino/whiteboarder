/**
 * @file HoughCircle_Demo.cpp
 * @brief Demo code for Hough Transform
 * @author OpenCV team
 */

#include "/Users/matthew/Downloads/opencv/modules/highgui/include/opencv2/highgui/highgui.hpp"
#include "/Users/matthew/Downloads/opencv/modules/imgproc/include/opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/**
 * @function main
 */
int main(int, char** argv)
{
   Mat src, src_gray;

   /// Read the image
   src = imread( argv[1], 1 );

   if( !src.data )
     { return -1; }

   /// Convert it to gray
    cvtColor( src, src_gray, COLOR_BGR2GRAY );

   /// Reduce the noise so we avoid false circle detection
    GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );

    vector<Vec3f> circles;

    //Canny(src_gray, src_gray,75,75*3,3);

   /// Apply the Hough Transform to find the circles

    //HoughCircles(src_gray,circles,CV_HOUGH_GRADIENT,1,80);
    //HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/8, 200, 100, 0, 0 );
    HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, 40, 10, 40, 5, 100 );

    std::cout << "Detections: " << circles.size();
   /// Draw the circles detected
    for( size_t i = 0; i < circles.size(); i++ )
    {
         Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
         int radius = cvRound(circles[i][2]);
         // circle center
         circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
         // circle outline
         circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
    }

   /// Show your results
    namedWindow( "Hough Circle Transform Demo", WINDOW_AUTOSIZE );
    imshow( "Hough Circle Transform Demo", src );

    waitKey(0);
    std::cout << std::endl << std::endl;
    return 0;
}
