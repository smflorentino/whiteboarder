import cv2
import numpy as np
filename = "C:\\Users\Matthew\My Documents\GitHub\whiteboarder\circlesarrowsnumbers.jpg"
img = cv2.imread(filename)
y,x,d=img.shape
img=cv2.resize(img,(x/4,y/4))



gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
circles =  cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1, 40, np.array([]), 100, 40, 5, 300)
if circles is not None:
            for c in circles[0]:
                    cv2.circle(img, (c[0],c[1]), c[2], (100,255,100),2)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,4,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
print "circles"

print circles

print "\ndots"
print dst


cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()


