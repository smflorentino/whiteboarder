import cv2
import numpy as np
import math
import os
import sys
from common import mosaic

from digits import *
from matplotlib import pyplot as plt

#Global Variables
corners = []
frame = None
img = None

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
  thinkness = 1
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

  #Determine which direction the arrow points inside the box
  def findDirection(self):
    #Divide the bound into two halves.
    rect1 = cv2.minAreaRect(getPointList(self.point1,self.point3,self.pointm1,self.pointm2))
    rect2 = cv2.minAreaRect(getPointList(self.point2,self.point4,self.pointm1,self.pointm2))
    r1Count = 0
    r2Count = 0
    #Check the count of each "corner" in each half.
    for corner in corners:
      (x,y) = i.ravel()
      if(cv2.pointPolygonTest(rect1,(x,y),False)==1.0):
        r1Count += 1
      if(cv2.pointPolygonTest(rect2,(x,y),False)==1.0):
        r2Count += 1
    if(r1Count > r2Count):
      return True
    if(r2Count > r1Count):
      return False

#Get the Bounding box for two circles. The arrow will be inside.
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

def getPointList(point1,point2,point3,point4):
  return np.array([[point1.x,point1.y],[point2.x,point2.y],[point3.x,point3.y],[point4.x,point4.y]])


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
def getCornerList():
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  corners = cv2.goodFeaturesToTrack(gray,50,0.05,0)
  corners = np.int0(corners)

  return corners;

#NUMBER DETECTION
def findNumbers():
    print "starting...."
    #blank = True
    #while blank:
    #    rval, frame = vc.read()
    #    if frame is not None:
    #        blank=False

    classifier_fn = 'digits_svm.dat'
    if not os.path.exists(classifier_fn):
        print '"%s" not found, run digits.py first' % classifier_fn
        return
    model = SVM()
    model.load(classifier_fn)

    stop=True
    while stop:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
        bin = cv2.medianBlur(bin, 3)
        contours, heirs = cv2.findContours( bin.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        try: heirs = heirs[0]
        except: heirs = []

        for cnt, heir in zip(contours, heirs):
            _, _, _, outer_i = heir
            if outer_i >= 0:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if not (16 <= h <= 64  and w <= 1.2*h):
                continue
            pad = max(h-w, 0)
            x, w = x-pad/2, w+pad
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0))

            bin_roi = bin[y:,x:][:h,:w]
            gray_roi = gray[y:,x:][:h,:w]

            m = bin_roi != 0
            if not 0.1 < m.mean() < 0.4:
                continue
            '''
            v_in, v_out = gray_roi[m], gray_roi[~m]
            if v_out.std() > 10.0:
                continue
            s = "%f, %f" % (abs(v_in.mean() - v_out.mean()), v_out.std())
            cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 0, 0), thickness = 1)
            '''

            s = 1.5*float(h)/SZ
            m = cv2.moments(bin_roi)
            c1 = np.float32([m['m10'], m['m01']]) / m['m00']
            c0 = np.float32([SZ/2, SZ/2])
            t = c1 - s*c0
            A = np.zeros((2, 3), np.float32)
            A[:,:2] = np.eye(2)*s
            A[:,2] = t
            bin_norm = cv2.warpAffine(bin_roi, A, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
            bin_norm = deskew(bin_norm)
            if x+w+SZ < frame.shape[1] and y+SZ < frame.shape[0]:
                frame[y:,x+w:][:SZ, :SZ] = bin_norm[...,np.newaxis]

            sample = preprocess_hog([bin_norm])
            digit = model.predict(sample)[0]
            cv2.putText(frame, '%d'%digit, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 0, 0), thickness = 1)
        stop=False


        cv2.imshow("Image Feed", frame)
        #cv2.imshow('bin', gray)
        #ch = cv2.waitKey(1)
        #if ch == 27:
        #    break



def process():
 
  #GET A NON BLANK IMAGE
  #blank = True
  #while blank:
  #    rval, img = vc.read()
  #    if img is not None:
  #     print "not empty"
  #      blank=False

  #CIRCLE DETECTION
  print "img",img
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  circles =  cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1, 40, np.array([]), 100, 40, 5, 300)
  if circles is None:
    return

  #CORNER DETECTION
  corners = getCornerList()


  for c in circles[0]:
    for d in circles[0]:  
        if(d[1]==c[1]):
            break
        cir = Circle(c[0],c[1],c[2])
        cir2 = Circle(d[0],d[1],d[2])
        b = findBox(cir,cir2,img)
        b.findDirection()
        #Draw Circle
        cv2.circle(img, (c[0],c[1]), c[2], (100,255,100),1)
        #Draw enter
        cv2.circle(img, (c[0],c[1]), 1, (100,100 ,255),1)
        #break
        cv2.line(img, (c[0],c[1]),(d[0],d[1]), (200,200,50))
        

  findNumbers()
  cv2.imshow("Image Feed",img)
        #break
        #r=30
        #theta=math.atan(-(d[0]-c[0])/(d[1]-c[1]))
        #test=np.array([0,r]);
        #points =getPointList(((c[0]+r*math.cos(theta)).astype(int),(c[1]+r*math.sin(theta)).astype(int)),((c[0]-r*math.cos(theta)).astype(int),(c[1]-r*math.sin(theta)).astype(int)),((d[0]+r*math.cos(theta)).astype(int),(d[1]+r*math.sin(theta)).astype(int)),((d[0]-r*math.cos(theta)).astype(int),(d[1]-r*math.sin(theta)).astype(int)))
        
        #rect = cv2.minAreaRect(points)
        #box = cv2.cv.BoxPoints(rect)
        #box = np.int0(box)
        #rect=((c[0],c[1]),(d[0],d[1]),((c[0]+r*math.cos(theta)).astype(int),(c[1]+r*math.sin(theta)).astype(int)),((c[0]-r*math.cos(theta)).astype(int),(c[1]-r*math.sin(theta)).astype(int)),((d[0]+r*math.cos(theta)).astype(int),(d[1]+r*math.sin(theta)).astype(int)),((d[0]-r*math.cos(theta)).astype(int),(d[1]-r*math.sin(theta)).astype(int)))

        #dotcount(points,dst,box)         
        #break

#IMAGE IMPORT
#filename = "C:\\Users\Matthew\My Documents\GitHub\whiteboarder\circlesarrowsnumbers.jpg"
#filename = "circlesarrowsnumbers.jpg"
#img = cv2.imread(filename)
#y,x,d=img.shape
#img=cv2.resize(img,(x/4,y/4))
#print x/4, y/4
#getCircles()

def main():
  cv2.namedWindow("Image Feed")
  vc = cv2.VideoCapture(0)
  global frame
  global img
  rval, frame = vc.read()

  while True:
    if frame is not None:
      #Show Live Image
      cv2.imshow("Image Feed", frame)
      img = frame
      print "img",img,"frame",frame
      #Process it instead
      process()
    if frame is None:
      print "empty"

    rval, frame = vc.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

#Call Main
main()


