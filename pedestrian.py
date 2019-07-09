import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
from imutils import paths
import argparse
import imutils
 


def def_contador():
    global contador
    contador = 0
"""
The parameters passed to the hog detector need to be played around
with to get optimum speed vs accuracy. This will always be a tradeoff.
"""
def detect_people(hog, image):
    global contador
    orig = image.copy()
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
        padding=(8, 8), scale=1.05)
 

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        contador += 1
        print("Persona %s",contador)
    # # show some information on the number of bounding boxes
    # filename = imagePath[imagePath.rfind("/") + 1:]
    # print("[INFO] {}: {} original boxes, {} after suppression".format(
    #     filename, len(rects), len(pick)))
 
def open_cam_and_detect(path):
    cap = cv2.VideoCapture(path)
    
    #Propiedades del video
    cap.set(3,800) #Width
    cap.set(4,600) #Height

    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
 
    if not cap.isOpened():
        raise AssertionError('Cap not opened')
 

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('ped_pocuro1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            raise AssertionError('Cap not opened')
 
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
        detect_people(hog, frame)
 
        # Write the frame into the file 'output.avi'
        out.write(frame)
        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    def_contador()
    open_cam_and_detect(0) # uses webcam
    open_cam_and_detect('temp.mp4') # uses file stored