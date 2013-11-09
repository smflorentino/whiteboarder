import cv2
import numpy as np
import math
#filename = "C:\\Users\Matthew\My Documents\GitHub\whiteboarder\circlesarrowsnumbers.jpg"
class Point:
  #x=None
  #y=None
  color = (200,200,50)
  thinkness=2

  def __init__(self,x,y):
    self.x = int(x)
    self.y = int(y)

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

'''
class Box:
  #Provide the smaller of the two radii and both centroids
  def __init__(self,center1, center2,radius):
    slope = Util.slope(center1,center2)
    perpSlope = Util.perpSlope(center1,center2)
'''
def slope(circle1, circle2):
  return (circle2[1]-circle1[1]) / (circle2[0] - circle1[0])

def perpSlope(circle1, circle2):
  return -(circle2[0]-circle1[0]) / (circle2[1] - circle1[1])

class Util:

    def slope(point1, point2):
      return (point2.y-point1.y) / (point2.x-point1.x)

    def perpSlope(point1, point2):
      return -(point2.x - point2.y) / (point2.y-point1.y)

def intersect(centroidX,centroidY,radius):
  return None






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

  #centroidLine = line(circle1[0],circle1[1],circle2[0],circle2[1])
  #centroidLine.draw(img)


def drawLine(line, img):
  x1 = line.point1
  x2 = x1.x
  print "points", line.point2.x, line.point1.x

  cv2.line(img, (line.point1.x,line.point1.y), (line.point2.x,line.point2.y), (200,200,50),4)





c1x,c1y,c2x,c2y=0,0,0,0
filename = "circlesarrowsnumbers.jpg"
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
       		#print (c[0],c[1]), (d[0],d[1])
          cv2.line(img, (c[0],c[1]), (d[0],d[1]), (200,200,50))
          c1x,c1y,c2x,c2y=c[0],c[1],d[0],d[1]
          findBox(c,d,img)
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
#print dst
if circles is not None:
            for c in circles[0]:
                    cv2.circle(img, (c[0],c[1]), c[2], (100,255,100),1)

cv2.imshow('dst',img)

print(c1x,c1y,c2x,c2y)
points = np.array([[100,200],[200,100],[400,300],[300,400]])
rect = cv2.minAreaRect(points)

box = cv2.cv.BoxPoints(rect)
box = np.int0(box)
cv2.drawContours(img,[box],0,(0,0,255),2)

#crop_image=img[c2x:c1x,c2y:c1y]
cv2.imshow("asd",img)


if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()



