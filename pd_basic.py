import numpy as np
import cv2
 
#cap = cv2.VideoCapture('../seq.mp4')

cap = cv2.VideoCapture(1)
 
#shrink the rectangle from default
def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        pad_w, pad_h = int(0.15 * w), int(0.05 * h)
        cv2.rectangle(img, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), thickness)
 
#default detector
def PD_default(filename):
    image = filename#cv2.imread(filename) #read image
    hog = cv2.HOGDescriptor() #derive HOG features
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) #setSVMDetector
 
    #pedestrian detection
    found, w = hog.detectMultiScale(image, hitThreshold = 0, winStride = (8,8), padding = (0, 0), scale = 1.05, finalThreshold = 5)
 
    draw_detections(image, found) #draw rectangles
 
    #write & save image
    cv2.imshow('original', image) #write image
    cv2.waitKey(1) #for keyboard binding
    #cv2.imwrite('test.jpg',image) # save image
 
while(cap.isOpened()):
    ret, frame = cap.read()
    PD_default(frame)
 
cap.release()
cv2.destroyAllWindows()
 
 
#print cap.grab()