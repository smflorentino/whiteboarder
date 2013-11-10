import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


#Utility Functions
def slope(circle1, circle2):
  return (circle2[1]-circle1[1]) / (circle2[0] - circle1[0])

def perpSlope(circle1, circle2):
  return -(circle2.x-circle1.x) / (circle2.y - circle1.y)

def midpoint(point1, point2):
  return Point((point1.x + point2.x)/2,(point1.y+point2.y)/2)

#A Single Point
class Point:
  color = (200,200,50)
  thinkness=5

  def __init__(self,x,y):
    self.x = int(x)
    self.y = int(y)

  def draw(self, img):
    print "xy",self.x,self.y
    cv2.circle(img,(self.x,self.y),4,255,-1)

#A line with two points.
class line:
  point1 = Point(0,0)
  point2 = Point(0,0)
  #def __init__(self,x1,x2,y1,y2):
  #  self.point1 = Point(x1,y1)
  #  self.point2 = Point(x2,y2)

  #See http://math.stackexchange.com/questions/9365/endpoint-of-a-line-knowing-slope-start-and-distance
  def __init__(self,x,y,m,r):
    self.point1 = Point(x,y)
    c = 1 / math.sqrt(1 + math.pow(m,2))
    s = m / math.sqrt(1 + math.pow(m,2))
    self.point2 = Point (x+r*c,y+r*s)

  def setPoints(self,point1,point2):
    self.point1 = point1
    self.point2 = point2

  def draw(self,img):
    cv2.line(img,(self.point1.x,self.point1.y),(self.point2.x,self.point2.y),Point.color,Point.thinkness)

  def drawEnds(self,img):
    self.point1.draw(img)
    self.point2.draw(img)

    #A line with two points.
class regLine:
  thinkness = 2
  color = (0,255,0)
  point1 = Point(0,0)
  point2 = Point(0,0)
  #def __init__(self,x1,x2,y1,y2):
  #  self.point1 = Point(x1,y1)
  #  self.point2 = Point(x2,y2)

  #See http://math.stackexchange.com/questions/9365/endpoint-of-a-line-knowing-slope-start-and-distance

  def __init__(self,point1,point2):
    self.point1 = point1
    self.point2 = point2

  def draw(self,img):
    cv2.line(img,(self.point1.x,self.point1.y),(self.point2.x,self.point2.y),regLine.color,regLine.thinkness)

  def drawEnds(self,img):
    self.point1.draw(img)
    self.point2.draw(img)



class Circle:
  def __init__(self,x,y,r):
    self.x=x
    self.y=y
    self.r=r

  def contains(self,x,y):
    return ( (x-self.x)^2 + (y-self.y)^2 ) < (self.r)^2

class Box:
  def __init__(self,point1,point2,point3,point4):
    self.point1 = point1
    self.point2 = point2
    self.point3 = point3
    self.point4 = point4
    self.pointm1 = midpoint(point1,point3)
    self.pointm2 = midpoint(point2,point4)
    self.side1 = regLine(point1,point2)
    self.side2 = regLine(point1,point3)
    self.side3 = regLine(point2,point4)
    self.side4 = regLine(point3,point4)
    self.side5 = regLine(self.pointm1,self.pointm2)

  def drawDots(self,img):
    self.point1.draw(img)
    self.point2.draw(img)
    self.point3.draw(img)
    self.point4.draw(img)
    self.pointm1.draw(img)
    self.pointm2.draw(img)

  def drawBox(self,img):
    self.side1.draw(img)
    self.side2.draw(img)
    self.side3.draw(img)
    self.side4.draw(img)
    self.side5.draw(img)

def findBox(circle1, circle2,img):
  #Get minimum radius of both circles

  radius1 = circle1.r
  radius2 = circle2.r
  radius =  min(radius1,radius2) + 10
  m = perpSlope(circle1,circle2)

  c1ctr = Point(circle1.x,circle1.y)
  c1ctr.draw(img)
  perpLine1a = line(c1ctr.x,c1ctr.y, m,radius)
  perpLine1a.drawEnds(img)
  
  c2ctr = Point(circle2.x,circle2.y)
  c2ctr.draw(img)
  perpLine2a = line(c2ctr.x,c2ctr.y,m,radius)
  perpLine2a.drawEnds(img)

  perpLine1b = line(c1ctr.x,c1ctr.y, m,-radius)
  perpLine1b.drawEnds(img)

  perpLine2b = line(c2ctr.x,c2ctr.y,m,-radius)
  perpLine2b.drawEnds(img)

  b = Box(perpLine1a.point2,perpLine1b.point2,perpLine2a.point2,perpLine2b.point2)
  b.drawBox(img)
  return b
  #perpLine1a.setPoints(perpLine1a.point2,perpLine1b.point2)
  #perpLine2a.setPoints(perpLine2b.point2,perpLine2b.point2)

  #perpLine1a.drawEnds(img)
  #perpLine1b.drawEnds(img)



def getPointList((x1,y1),(x2,y2),(x3,y3),(x4,y4)):
  return np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])


#ARROW DETECTION
def dotcount(points,dst,box):
  return
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


#CORNER DETECTION
def getCornerList(img):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  corners = cv2.goodFeaturesToTrack(gray,50,0.05,0)
  corners = np.int0(corners)

  return corners;


#CIRCLE DETECTION
def getCircles():
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  circles =  cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1, 40, np.array([]), 100, 40, 5, 300)
  #print "List", len(circles[0])
  #first=True

  circles=circles[0]
#EDGE DETECTION
    
  gray = np.float32(gray)
  dst = cv2.cornerHarris(gray,4,3,0.04)
#result is dilated for marking the corners, not important
  dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
  img[dst>0.01*dst.max()]=[0,0,255]

  circleList = []
  
  
  for i in range (0,circles.shape[0]):
    for j in range(0,circles.shape[0]):  
        if(i!=j):
          cir = Circle(circles[i],c[1],c[2])
          cir2 = Circle(d[0],d[1],d[2])
          findBox(cir,cir2,img)
          circleList.append(cir)
          break
          cv2.line(img, (c[0],c[1]),(d[0],d[1]), (200,200,50))
          r=30
          theta=math.atan(-(circles[i,0]-circles[i,0])/(circles[i,1]-circles[i,1]))
          test=np.array([0,r]);
          points =getPointList(((circles[i,0]+r*math.cos(theta)).astype(int),(circles[i,1]+r*math.sin(theta)).astype(int)),((circles[i,0]-r*math.cos(theta)).astype(int),(circles[i,1]-r*math.sin(theta)).astype(int)),((circles[i,0]+r*math.cos(theta)).astype(int),(circles[i,1]+r*math.sin(theta)).astype(int)),((circles[i,0]-r*math.cos(theta)).astype(int),(circles[i,1]-r*math.sin(theta)).astype(int)))
          
          rect = cv2.minAreaRect(points)
          box = cv2.cv.BoxPoints(rect)
          box = np.int0(box)
          #rect=((c[0],c[1]),(d[0],d[1]),((c[0]+r*math.cos(theta)).astype(int),(c[1]+r*math.sin(theta)).astype(int)),((c[0]-r*math.cos(theta)).astype(int),(c[1]-r*math.sin(theta)).astype(int)),((d[0]+r*math.cos(theta)).astype(int),(d[1]+r*math.sin(theta)).astype(int)),((d[0]-r*math.cos(theta)).astype(int),(d[1]-r*math.sin(theta)).astype(int)))

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
  #if circles is not None:
  #          for c in circles[0]:
                    #cv2.circle(img, (c[0],c[1]), c[2], (100,255,100),-1)

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


