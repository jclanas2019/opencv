import numpy as np
import cv2


cap = cv2.VideoCapture('detector-movimiento-opencv.mp4')


fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=200, nmixtures=5, backgroundRatio=0.7, noiseSigma=0)


cv2.ocl.setUseOpenCL(False)

while(1):

	ret, frame = cap.read()


	if not ret:
		break


	fgmask = fgbg.apply(frame)


	contornosimg = fgmask.copy()


	im, contornos, hierarchy = cv2.findContours(contornosimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


	for c in contornos:

		if cv2.contourArea(c) < 500:
			continue

		(x, y, w, h) = cv2.boundingRect(c)

		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	cv2.imshow('Camara',frame)
	cv2.imshow('Umbral',fgmask)
	cv2.imshow('Contornos',contornosimg)


	k = cv2.waitKey(30) & 0xff
	if k == ord("s"):
		break

cap.release()
cv2.destroyAllWindows()