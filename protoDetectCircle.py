import numpy
import os
import cv2
from matplotlib import pyplot as plt
print "fuck"
os.chdir("C:\\Users\Matthew\My Documents\GitHub\whiteboarder")
img=cv2.imread("circlesarrowsnumbers.jpg",0)
#img=cv2.resize(bigimg,(600,400))
#img=cv2.bilateralFilter(img,9,25,25)
img=cv2.resize(img,(1000,650))

circles =  cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, 40, numpy.array([]), 100, 40, 5, 300)
#Draw the circles detected
#print circles

#print circles
if circles is not None:
            for c in circles[0]:
                    cv2.circle(img, (c[0],c[1]), c[2], (100,255,100),2)
#lines=cv.HoughLines(img,numpy.array([]),cv2.cv.CV_HOUGH_PROBABILISTIC,10,2,50,20,20
                    #cv2.Canny(img,20,80)

dst = cv2.cornerHarris(img,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',img)

cv2.imshow("SHOW ME THE CIRCLES AND LINES AND SHIT",img)
cv2.waitKey(0);
