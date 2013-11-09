import numpy
import os
import cv2
from matplotlib import pyplot as plt
print "fuck"
os.chdir("C:\\Users\Matthew\My Documents\GitHub\whiteboarder")
img=cv2.imread("circlesarrowsnumbers.jpg",0)
y,x=img.shape
print x,y
img=cv2.resize(img,(x/4,y/4))

circles =  cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, 40, numpy.array([]), 100, 40, 5, 300)
#Draw the circles detected
#print circles

#print circles
if circles is not None:
            for c in circles[0]:
                    cv2.circle(img, (c[0],c[1]), c[2], (100,255,100),2)

cv2.imshow("SHOW ME THE CIRCLES AND LINES AND SHIT",img)
cv2.waitKey(0);
