#Ejemplo de deteccion facial con OpenCV y Python
#Por Glare
#www.robologs.net
 
import numpy as np
import cv2

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

#cargamos la plantilla e inicializamos la webcam:
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# cv2.namedWindow('image',WINDOW_NORMAL)
cap = cv2.VideoCapture("expo2.avi")
cv2.namedWindow("main", cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow('main', 1280, 720)
#cv2.resizeWindow('image', 600,600)

 
while(True):
    #leemos un frame y lo guardamos
    ret, img = cap.read()
    frame75 = rescale_frame(img, percent=75)
 
    #convertimos la imagen a blanco y negro
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    #buscamos las coordenadas de los rostros (si los hay) y
    #guardamos su posicion
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
 
    #Dibujamos un rectangulo en las coordenadas de cada rostro
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(125,255,0),2)


    #cv2.imshow('frame75', frame75)
    #Mostramos la imagen
    cv2.imshow('img',img)
     
    #con la tecla 'q' salimos del programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2-destroyAllWindows()


