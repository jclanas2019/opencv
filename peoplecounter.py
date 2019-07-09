from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2
import requests
import time
import argparse
import time
 
'''
Usage:
 
python peopleCounter.py -i PATH_TO_IMAGE  # Reads and detect people in a single local stored image
python peopleCounter.py -c  # Attempts to detect people using webcam
 
IMPORTANT: This example is given AS IT IS without any warranty
 
Made by: Jose Garcia
 
'''
 
URL_EDUCATIONAL = "http://things.ubidots.com"
URL_INDUSTRIAL = "http://industrial.api.ubidots.com"
INDUSTRIAL_USER = True  # Set this to False if you are an educational user
TOKEN = "...."  # Put here your Ubidots TOKEN
DEVICE = "detector"  # Device where will be stored the result
VARIABLE = "people"  # Variable where will be stored the result
 
# Opencv pre-trained SVM with HOG people features
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
 
 
def detector(image):
    '''
    @image is a numpy array
    '''
 
    image = imutils.resize(image, width=min(400, image.shape[1]))
    clone = image.copy()
 
    (rects, weights) = HOGCV.detectMultiScale(image, winStride=(8, 8),
                                              padding=(32, 32), scale=1.05)
 
    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
    # Applies non-max supression from imutils package to kick-off overlapped
    # boxes
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    result = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 
    return result
 
 
def buildPayload(variable, value, context):
    return {variable: {"value": value, "context": context}}
 
 
def sendToUbidots(token, device, variable, value, context={}, industrial=True):
    # Builds the endpoint
    url = URL_INDUSTRIAL if industrial else URL_EDUCATIONAL
    url = "{}/api/v1.6/devices/{}".format(url, device)
 
    payload = buildPayload(variable, value, context)
    headers = {"X-Auth-Token": token, "Content-Type": "application/json"}
 
    attempts = 0
    status = 400
 
    while status >= 400 and attempts <= 5:
        req = requests.post(url=url, headers=headers, json=payload)
        status = req.status_code
        attempts += 1
        time.sleep(1)
 
    return req
 
 
def argsParser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default=None,
                    help="path to image test file directory")
    ap.add_argument("-c", "--camera", default=False,
                    help="Set as true if you wish to use the camera")
    args = vars(ap.parse_args())
 
    return args
 
 
def localDetect(image_path):
    result = []
    image = cv2.imread(image_path)
    if len(image) <= 0:
        print("[ERROR] could not read your local image")
        return result
    print("[INFO] Detecting people")
    result = detector(image)
 
    # shows the result
    for (xA, yA, xB, yB) in result:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
 
    cv2.imshow("result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
    return result
 
 
def cameraDetect(token, device, variable, sample_time=5):
 
    cap = cv2.VideoCapture(1)
    init = time.time()
 
    # Allowed sample time for Ubidots is 1 dot/second
    if sample_time < 1:
        sample_time = 1
 
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
 
        result = detector(frame.copy())
 
        # shows the result
        for (xA, yA, xB, yB) in result:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        cv2.imshow('frame', frame)
 
        # Sends results
        if time.time() - init >= sample_time:
            print("[INFO] Sending actual frame results")
            sendToUbidots(token, device, variable, len(result))
            init = time.time()
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
 
 
def detectPeople(args):
    image_path = args["image"]
    camera = True if str(args["camera"]) == 'true' else False
 
    # Routine to read local image
    if image_path != None and not camera:
        print("[INFO] Image path provided, attempting to read image")
        result = localDetect(image_path)
        print("[INFO] sending results")
        req = sendToUbidots(TOKEN, DEVICE, VARIABLE, len(result))
        if req.status_code >= 400:
            print("[ERROR] Could not send data to Ubidots")
            return req
 
    # Routine to read images from webcam
    if camera:
        print("[INFO] reading camera images")
        cameraDetect(TOKEN, DEVICE, VARIABLE)
 
 
def main():
    args = argsParser()
    print("[INFO] reading camera images")
    cameraDetect(TOKEN, DEVICE, VARIABLE)
    detectPeople(args)
 
 
if __name__ == '__main__':
    main()