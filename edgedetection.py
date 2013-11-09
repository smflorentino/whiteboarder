import cv2
import numpy as np
import math
filename = "C:\\Users\Matthew\My Documents\GitHub\whiteboarder\circlesarrowsnumbers.jpg"
img = cv2.imread(filename)
y,x,d=img.shape
img=cv2.resize(img,(x/4,y/4))

#create duplicate for node processing
#img2 = cv2.imread(filename)
#img2=cv2.resize(img2,(x/4,y/4))



gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
circles =  cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1, 40, np.array([]), 100, 40, 5, 300)
first=True
for c in circles[0]:
    for d in circles[0]:
        if(c[0]==d[0] and d[1]==c[1]):
            break
        #line=cv2.fitline([(c[0],c[1]),(d[0],d[1])],CV_DIST_L2,0,.01,.01)
        if (first):

            cv2.line(img, (c[0],c[1]), (d[0],d[1]), (200,200,50))
       # print c[0]
       # print d[0]
        r=min(c[2],d[2])
        theta=math.atan(-(d[0]-c[0])/(d[1]-c[1]))
        
       
     #   print min(c[2],d[2])
        r=((c[0],c[1]),(d[0],d[1]),(c[0]+r*math.cos(theta),c[1]+r*math.sin(theta)),(c[0],c[1]),(d[0],d[1]),(d[0],d[1]))
      #  print r
        if (first):
            cv2.line(img, r[0],r[3], (200,200,50))
        first=False
       

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,4,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
print "circles", circles.size, circles.shape

#print circles

#print "\ndots", dst.size,dst.shape
print dst
if circles is not None:
            for c in circles[0]:
                    cv2.circle(img, (c[0],c[1]), c[2], (100,255,100),1)

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()


