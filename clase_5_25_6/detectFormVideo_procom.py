import cv2
import numpy as np

############################################
# Test Video 1
if(0):
    frame_width = 1280; frame_height = 720
    video = cv2.VideoCapture("clase_5_25_6/videos/videoForm.mp4")
############################################

############################################
# Test Video 2
if(0):
    frame_width = 1920; frame_height = 1080
    video = cv2.VideoCapture("clase_5_25_6/videos/videoForm2.mp4")
############################################

############################################
# Test Video 2
if(1):
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
#VideoWriter_fourcc() formato para guardar el video 
# AVI
#videoOut = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)
############################################
#Procesamiento

while(video.isOpened()):
    #captura un frame y elimina ese frame del video
    ret, img = video.read()  #img = frame, ret = estado valido del frame si es false llego al final del archivo

    if ret == False:
        break
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ############################################
    # Pre-process Frame
    _, thresh = cv2.threshold(imgGray,240,255,cv2.THRESH_BINARY)
    cv2.imshow("Threshold", thresh)
    countours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    ############################################

    ############################################
    # Countour Analysis
    for countour in countours:
        approx = cv2.approxPolyDP(countour,0.01*cv2.arcLength(countour,True),True)
        cv2.drawContours(img,[approx],0,(255,0,0),2)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        if   len(approx)==3:
            cv2.putText(img,"Triangle", (x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255))
        elif len(approx)==4:
            x,y,w,h = cv2.boundingRect(approx)
            aspectRatio = float(w)/h

            if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                cv2.putText(img,"Square", (x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255))
            else:
                cv2.putText(img,"Rectangule", (x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255))
        elif len(approx)==5:
            cv2.putText(img,"Pentagon", (x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255))
        elif len(approx)==10:
            cv2.putText(img,"Rectangule", (x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255))
        else:
            cv2.putText(img,"Circle", (x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255))
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