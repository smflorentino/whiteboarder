import numpy
import os
import cv2
from matplotlib import pyplot as plt
print "fuck"
os.chdir("C:\\Users\Matthew\My Documents\GitHub\whiteboarder")
img=cv2.imread("circles.jpg",0)
#img=cv2.resize(bigimg,(600,400))
#img=cv2.bilateralFilter(img,9,25,25)
circles =  cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, 40, numpy.array([]), 100, 20, 5, 100)
#Draw the circles detected
print circles
if circles is not None:
            for c in circles[0]:
                    cv2.circle(img, (c[0],c[1]), c[2], (0,255,0),2)
cv2.imshow("SHOW ME THE CIRCLES",img)
cv2.waitKey(0);
