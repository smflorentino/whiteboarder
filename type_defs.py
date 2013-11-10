class Circle:
  def __init__(self,x,y,r):
    self.x=x
    self.y=y
    self.r=r

  def contains(self,x,y):
    return ( (x-self.x)^2 + (y-self.y)^2 ) < (self.r)^2

'''
         #Create Circle Objects (x,y,r)
        cir = Circle(c[0],c[1],c[2])
        circleList.append(cir)
'''