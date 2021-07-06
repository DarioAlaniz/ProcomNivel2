import cv2
import numpy as np
from shapeDetection import ShapeDetector
############################################
# Test Video 1
if(0):
    frame_width = 1280; frame_height = 720
    video = cv2.VideoCapture("clase_5_25_6/videos/videoForm.mp4")
############################################

############################################
# Test Video 2
if(1):
    frame_width = 640; frame_height = 360
    video = cv2.VideoCapture("clase_5_25_6/videos/videoForm2.mp4")
############################################

############################################
# Test Video 2
if(0):
    frame_width = 640; frame_height = 360
    video = cv2.VideoCapture("clase_5_25_6/videos/videoplayback.mp4")
############################################

############################################
# Turn On camera
if(0):
    frame_width = 1920; frame_height = 1080
    video = cv2.VideoCapture(0)
############################################

fps      = 30               #frecuencia de reproduccin de salida
size     = (frame_width,frame_height)
pathOut  = 'clase_5_25_6/videos/outpy.mp4' #path para guardar el video de salida

# controlo la ventana del video
#Variable used to control the speed of reading the video
ControlSpeedVar = 100  #Lowest: 1 - Highest:100
HiSpeed = 100

############################################
# MP4
videoOut = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

sd = ShapeDetector()

while(video.isOpened()):
    #captura un frame y elimina ese frame del video
    ret, img = video.read()  #img = frame, ret = estado valido del frame si es false llego al final del archivo
    img = cv2.resize(img,(640,360))
    if ret == False:
        break
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgGray = cv2.resize(imgGray,(648,360))

    ############################################
    # Pre-process Frame
    # gauss = cv2.GaussianBlur(imgGray,(11,11),0)
    _, thresh = cv2.threshold(imgGray,240,255,cv2.THRESH_BINARY)
    cv2.imshow("Threshold", thresh)
    countours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    ############################################

    ############################################
    # Countour Analysis
    for countour in countours:
        shape,approx = sd.detect(countour)
        x,y,w,h = cv2.boundingRect(approx)
        cv2.drawContours(img,[approx],0,(255,0,0),2)
        #coordenads para escribir el texto    
        M = cv2.moments(countour) #momento de una imagen: promedio ponderado de la intencidad de pixeles 
        #Calculo en centro de una imagen
        # cx = int(M["m10"]/M["m00"])
        # cy = int(M["m01"]/M["m00"])
        cx = approx.ravel()[0]
        cy = approx.ravel()[1]
        cv2.putText(img,shape, (cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0))
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    ############################################

    videoOut.write(img)

    cv2.imshow('image', img)

    ############################################
    # Close script and speed
    if cv2.waitKey(2) & 0xFF == ord('q'):
        video.release()         #cierra los archivos
        videoOut.release()
        cv2.destroyAllWindows()
        break

    cv2.waitKey(HiSpeed-ControlSpeedVar+1) #controla la velocidad de reproduccion
    ############################################

cv2.destroyAllWindows()