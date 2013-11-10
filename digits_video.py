#!/usr/bin/env python

import numpy as np
import cv2
import os
import sys
import time
from common import mosaic

from digits import *

def process(vc,frame):
    print "starting...."
    blank = True
    while blank:
        rval, frame = vc.read()
        if frame is not None:
            blank=False

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


        cv2.imshow('frame', frame)
        cv2.imshow('bin', gray)
        ch = cv2.waitKey(1)
        if ch == 27:
            break

def process2(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    circles =  cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1, 40, np.array([]), 100, 40, 5, 300)
    if circles is not None:
        for c in circles[0]:
            #Draw Circle
            cv2.circle(img, (c[0],c[1]), c[2], (100,255,100),2)
            #Draw enter
            cv2.circle(img, (c[0],c[1]), 1, (100,100 ,255),2)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,4,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]
    #findarrows(circles,img)
    cv2.imshow("Image Feed",img)

s = ""

import thread
import time

# Define a function for the thread
def display_preview(threadName, delay,vc):
   while True:
      time.sleep(delay)
      rval,frame = vc.read()
      cv2.imshow("Image Feed",frame)
      print "%s: %s" % ( threadName, time.ctime(time.time()) )

def main2():
    cv2.namedWindow("Image Feed")
    vc = cv2.VideoCapture(0)
    rval, frame = vc.read()
    Wait = True
    try:
        thread.start_new_thread( display_preview, ("Thread-1", 0,vc) )
    except:
        print "Error: unable to start thread"
    while Wait:
        s = raw_input()
        cv2.namedWindow("Image Feed")
        vc = cv2.VideoCapture(0)
        rval, frame = vc.read()
        cv2.imshow("Current Image,frame")
        Wait = False

def main():
        cv2.namedWindow("Image Feed")
        vc = cv2.VideoCapture(0)

        rval, frame = vc.read()

        empty = 0
        while True:
            if frame is not None:
                #Show Live Image
                #cv2.imshow("preview", frame)
                #Process it instead
                #cv2.imshow("Current Image",frame)
                process(vc,frame)
            if frame is None:
                empty = empty + 1
                print "Empty", empty
            rval, frame = vc.read()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                process(vc,frame)
                cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



#Call Main
main()
