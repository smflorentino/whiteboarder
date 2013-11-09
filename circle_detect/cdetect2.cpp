/**
 * @file HoughCircle_Demo.cpp
 * @brief Demo code for Hough Transform
 * @author OpenCV team
 */

#include "/Users/scottflo/Downloads/opencv-2.4.6.1/modules/highgui/include/opencv2/highgui/highgui.hpp"
#include "/Users/scottflo/Downloads/opencv-2.4.6.1/modules/imgproc/include/opencv2/imgproc/imgproc.hpp"
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

    //Canny
    //Canny(src_gray, src_gray,60,60*3,3);

   /// Apply the Hough Transform to find the circles

    //HoughCircles(src_gray,circles,CV_HOUGH_GRADIENT,1,80);
    //HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/8, 200, 100, 0, 0 );
    
    //Detect Circles

    HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, 40, 10, 45, 5, 100 );

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

    Canny(src_gray, src_gray,60,60*3,3);
    vector<Vec4i> lines;
    HoughLinesP( src_gray, lines, 1, CV_PI/180, 80, 30, 10 );
    for( size_t i = 0; i < lines.size(); i++ )
    {
        line( src, Point(lines[i][0], lines[i][1]),
            Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 3, 8 );
    }

    //Canny(src,src,60,60*3,3);

   /// Show your results
    namedWindow( "Hough Circle Transform Demo", WINDOW_AUTOSIZE );
    imshow( "Hough Circle Transform Demo", src );

    waitKey(0);
    std::cout << std::endl << std::endl;
    return 0;
}
