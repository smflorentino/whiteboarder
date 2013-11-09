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
   resize(src, src,Size(),.33,.33);

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
    
    /*
    Parameters: 
image – 8-bit, single-channel binary source image. The image may be modified by the function.
lines – Output vector of lines. Each line is represented by a 4-element vector  (x_1, y_1, x_2, y_2) , where  (x_1,y_1) and  (x_2, y_2) are the ending points of each detected line segment.
rho – Distance resolution of the accumulator in pixels.
theta – Angle resolution of the accumulator in radians.
threshold – Accumulator threshold parameter. Only those lines are returned that get enough votes ( >\texttt{threshold} ).
minLineLength – Minimum line length. Line segments shorter than that are rejected.
maxLineGap – Maximum allowed gap between points on the same line to link them.
    */
    HoughLinesP( src_gray, lines, 1, CV_PI/180, 80, 10, 2 );
    /*for( size_t i = 0; i < lines.size(); i++ )
    {
        line( src_gray, Point(lines[i][0], lines[i][1]),
            Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 3, 8 );
    }*/

    //Canny(src,src,60,60*3,3);

   /// Show your results
    namedWindow( "Hough Circle Transform Demo", WINDOW_AUTOSIZE );
    imshow( "Hough Circle Transform Demo", src );

    namedWindow("Canny Results", WINDOW_AUTOSIZE);
    imshow("Hough Line Transform Demo", src_gray);

    waitKey(0);
    std::cout << std::endl << std::endl;
    return 0;
}
