import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


#Utility Functions
def slope(circle1, circle2):
  return (circle2[1]-circle1[1]) / (circle2[0] - circle1[0])

def perpSlope(circle1, circle2):
  return -(circle2[0]-circle1[0]) / (circle2[1] - circle1[1])

#A Single Point
class Point:
  color = (200,200,50)
  thinkness=2

  def __init__(self,x,y):
    self.x = int(x)
    self.y = int(y)

#A line with two points.
class line:
  point1 = Point(0,0)
  point2 = Point(0,0)

  def __init__(self,x1,x2,y1,y2):
    self.point1 = Point(x1,y1)
    self.point2 = Point(x2,y2)

  #See http://math.stackexchange.com/questions/9365/endpoint-of-a-line-knowing-slope-start-and-distance
  def __init__(self,x,y,m,r):
    self.point1 = Point(x,y)
    c = 1 / math.sqrt(1 + math.pow(m,2))
    s = m / math.sqrt(1 + math.pow(m,2))
    self.point2 = Point (x+r*c,y+r*s)

  def draw(self,img):
    cv2.line(img,(self.point1.x,self.point1.y),(self.point2.x,self.point2.y),Point.color,Point.thinkness)

def findBox(circle1, circle2,img):
  #Get minimum radius of both circles

  radius1 = circle1[2]
  radius2 = circle2[2]
  radius =  min(radius1,radius2)
  m = perpSlope(circle1,circle2)

  c1ctr = Point(circle1[0],circle1[1])
  perpLine1a = line(c1ctr.x,c1ctr.y, m,radius)
  perpLine1a.draw(img)
  
  c2ctr = Point(circle2[0],circle2[1])
  perpLine2a = line(c2ctr.x,c2ctr.y,m,radius)
  perpLine2a.draw(img)

  perpLine1b = line(c1ctr.x,c1ctr.y, m,-radius)
  perpLine1b.draw(img)

  perpLine2b = line(c2ctr.x,c2ctr.y,m,-radius)
  perpLine2b.draw(img)

def getPointList((x1,y1),(x2,y2),(x3,y3),(x4,y4)):
  return np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])


#ARROW DETECTION
def dotcount(points,dst,box):
  #print points
  xmin=min(min(points[0,0],points[1,0]),min(points[2,0],points[3,0]))
  xmax=max(max(points[0,0],points[1,0]),max(points[2,0],points[3,0]))
  ymin=min(min(points[0,1],points[1,1]),min(points[2,1],points[3,1]))
  ymax=max(max(points[0,1],points[1,1]),max(points[2,1],points[3,1]))
  count=0
  h,w,d=img.shape
  dm=dst.max()
  print xmin,xmax,ymin,ymax
  for x in range(xmin,xmax):
    #print "x",x, "xmax", xmax,"ymax",ymax
    #print box
    for y in range(ymin,ymax):
     # print y,x,dst[y,x],cv2.pointPolygonTest(points,(y,x),False)
      if (dst[y,x]>.01*dm)& (cv2.pointPolygonTest(box,(y,x),False)==1.0):
            count+=1
            
  print "count",count



#CIRCLE DETECTION
def getCircles():
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  circles =  cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1, 40, np.array([]), 100, 40, 5, 300)
  #print "List", len(circles[0])
  #first=True


#EDGE DETECTION
    
  gray = np.float32(gray)
  dst = cv2.cornerHarris(gray,4,3,0.04)
#result is dilated for marking the corners, not important
  dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
  img[dst>0.01*dst.max()]=[0,0,255]


  
  for c in circles[0]:
    for d in circles[0]:  
        if(d[1]==c[1]):
            break
        cv2.line(img, (c[0],c[1]),(d[0],d[1]), (200,200,50))
        r=30
        theta=math.atan(-(d[0]-c[0])/(d[1]-c[1]))
        test=np.array([0,r]);
        points =getPointList(((c[0]+r*math.cos(theta)).astype(int),(c[1]+r*math.sin(theta)).astype(int)),((c[0]-r*math.cos(theta)).astype(int),(c[1]-r*math.sin(theta)).astype(int)),((d[0]+r*math.cos(theta)).astype(int),(d[1]+r*math.sin(theta)).astype(int)),((d[0]-r*math.cos(theta)).astype(int),(d[1]-r*math.sin(theta)).astype(int)))
        
        rect = cv2.minAreaRect(points)
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        rect=((c[0],c[1]),(d[0],d[1]),((c[0]+r*math.cos(theta)).astype(int),(c[1]+r*math.sin(theta)).astype(int)),((c[0]-r*math.cos(theta)).astype(int),(c[1]-r*math.sin(theta)).astype(int)),((d[0]+r*math.cos(theta)).astype(int),(d[1]+r*math.sin(theta)).astype(int)),((d[0]-r*math.cos(theta)).astype(int),(d[1]-r*math.sin(theta)).astype(int)))

        dotcount(points,dst,box)          

       # cv2.line(img, rect[0],rect[1], (0,200,250))
        #cv2.line(img,rect[2], rect[3], (0,100,250),2)
       # cv2.line(img,rect[4], rect[5], (0,100,250),2)
       # if first:
         # print "shit is about to have happened, man"
          
       # cv2.drawContours(img,[box],0,(0,0,255),2)

          #bitmask(img,theta,rect)
          #first=False




#PAINTING
  if circles is not None:
            for c in circles[0]:
                    cv2.circle(img, (c[0],c[1]), c[2], (100,255,100),-1)

  cv2.imshow('dst',img)

  if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()


#IMAGE IMPORT
#filename = "C:\\Users\Matthew\My Documents\GitHub\whiteboarder\circlesarrowsnumbers.jpg"
filename = "circlesarrowsnumbers.jpg"
img = cv2.imread(filename)
y,x,d=img.shape
img=cv2.resize(img,(x/4,y/4))
print x/4, y/4
getCircles()


