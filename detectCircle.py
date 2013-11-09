import cv2
import cv
import numpy as np


img = cv2.imread('test2.jpg')
blank = cv2.imread('blank.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(gray,cv.CV_HOUGH_GRADIENT,1,50)
circles = circles[0]
print circles
for c in circles:
    center = (c[0],c[1])
    radius = c[2]
    print center
    print radius
    cv2.circle(blank,center,radius,(255,0,0))

cv2.circle(img,(100,200),50,(255,0,0))
cv.NamedWindow("result")
cv.ShowImage("result",cv.fromarray(blank))
cv.WaitKey(0)
