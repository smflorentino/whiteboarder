import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

#IMAGE IMPORT
filename = "C:\\Users\Matthew\My Documents\GitHub\whiteboarder\circlesarrowsnumbers.jpg"
img = cv2.imread(filename)
y,x,d=img.shape
print x

img=cv2.resize(img,(x/4,y/4))
print x/4

#ARROW DETECTION
def bitmask(img,mask):
   return 0

#CIRCLE DETECTION

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
circles =  cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1, 40, np.array([]), 100, 40, 5, 300)
for c in circles[0]:
    for d in circles[0]:
        
        if(c[0]==d[0] and d[1]==c[1]):
            break
        #cv2.line(img, (c[0],c[1]),(d[0],d[1]), (200,200,50))
   
        r=min(c[2],d[2])//1
        theta=math.atan(-(d[0]-c[0])/(d[1]-c[1]))
        test=np.array([0,r]);
        rect=((c[0],c[1]),(d[0],d[1]),((c[0]+r*math.cos(theta)).astype(int),(c[1]+r*math.sin(theta)).astype(int)),((c[0]-r*math.cos(theta)).astype(int),(c[1]-r*math.sin(theta)).astype(int)),((d[0]+r*math.cos(theta)).astype(int),(d[1]+r*math.sin(theta)).astype(int)),((d[0]-r*math.cos(theta)).astype(int),(d[1]-r*math.sin(theta)).astype(int)))
       # cv2.line(img, rect[0],rect[1], (0,200,250))
        cv2.line(img,rect[2], rect[3], (0,100,250),2)
        cv2.line(img,rect[4], rect[5], (0,100,250),2)

       
#EDGE DETECTION
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,4,3,0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

#PAINTING
if circles is not None:
            for c in circles[0]:
                    cv2.circle(img, (c[0],c[1]), c[2], (100,255,100),-1)

cv2.imshow('dst',img)
cv2.waitKey(0)



