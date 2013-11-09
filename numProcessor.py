import numpy as np
import os
import cv2
import Image
print "This will generate values for the structures"

""" Take the inner square from the starting node and process the resulting sub image.
    For empty nodes, an empty node will be created.  Number values will be supported
    and a technique for generating values from handwritten letters will be used below
    Upon process of this subrectangle, the next node will be found. """

#os.chdir("C:\\Users\Matthew\My Documents\GitHub\whiteboarder")
img = cv2.imread("valueGen.jpg") #for testing
#def numberParser():
   #may or may not need 

def numberSlicer(img):
    arr = np.array(Image.open(img))
    step = 2**a
    for j in range(2**l):
        for i in range(2**l):
            block = arr[j * step:(j + 1) * step, i * step:(i + 1) * step]
            print "Hey"

numberSlicer(img)
