import cv2
import numpy as np
# from shapeDetection import ShapeDetector
from shapeDetection import ShapeDetector
import fileKernels as filter
############################################
# Test Video 1
if(0):
    frame_width = 1280; frame_height = 720
    video = cv2.VideoCapture("clase_5_25_6/videos/videoForm.mp4")
############################################

############################################
# Test Video 2
if(0):
    frame_width = 640; frame_height = 360
    video = cv2.VideoCapture("clase_5_25_6/videos/videoForm2.mp4")
if(0):
    limite = 240
    area_up = 190476
    area_low = 1170
    enable_filter_gauss = 0
    size_gauss = 5
    enable_filter_canny = 0
    up_canny = 200
    low_canny = 55
    enable_filter_kernel = 0
    tipe_kernel = 5  
############################################

############################################
# Test Video 2
if(0):
    frame_width = 640; frame_height = 360
    video = cv2.VideoCapture("clase_5_25_6/videos/videoplayback.mp4")
if(0):
    limite = 217
    area_up = 27211
    area_low = 544
    enable_filter_gauss = 0
    size_gauss = 5
    enable_filter_canny = 0
    up_canny = 200
    low_canny = 55
    enable_filter_kernel = 0
    tipe_kernel = 5   
############################################

############################################
# Turn On camera
if(1):
    frame_width = 1920; frame_height = 1080
    video = cv2.VideoCapture(0)
#http://geoplana.blogspot.com/2008/09/figuras-geomtricas-planas-o-polgonos.html imagen tomada por la camara del celular
if(0):
    limite = 170
    area_up = 38721
    area_low = 295
    enable_filter_gauss = 0
    size_gauss = 5
    enable_filter_canny = 0
    up_canny = 200
    low_canny = 55
    enable_filter_kernel = 0
    tipe_kernel = 5 
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
############################################
#Create trackbars
def nothing(x):
    pass
if(1):
    nameWindows = 'Parametros'
    # Select_thres= '0 Threshold: OFF \n1 : Threshold: ON'
    lim_thres = 'lim_thresh'
    area_limit_up = 'upper limit of area'
    area_limit_low = 'lower limit of area'
    Select_Gauss = '0 Gauss : OFF  \n1 : Gauss : ON'
    Select_Canny = '0 Canny : OFF  \n1 : Canny : ON'
    canny_up = 'upper canny'
    canny_low='lower canny'
    Select_Kernel = '0 Kernel : OFF \n1 : Kernel: ON'
    tipe_of_kernel='tipe of kernel'

    cv2.namedWindow(nameWindows)
    cv2.resizeWindow(nameWindows,720,420)

    # cv2.createTrackbar(Select_thres,nameWindows,0,1,nothing)
    cv2.createTrackbar(lim_thres,nameWindows,0,255,nothing)
    cv2.createTrackbar(area_limit_up,nameWindows,0,200000,nothing)
    cv2.createTrackbar(area_limit_low,nameWindows,0,2000,nothing)
    cv2.createTrackbar(Select_Gauss,nameWindows,0,1,nothing)
    cv2.createTrackbar('Size Gauss',nameWindows,1,50,nothing)
    cv2.createTrackbar(Select_Canny,nameWindows,0,1,nothing)
    cv2.createTrackbar(canny_up,nameWindows,0,255,nothing)
    cv2.createTrackbar(canny_low,nameWindows,0,255,nothing)
    cv2.createTrackbar(Select_Kernel,nameWindows,0,1,nothing)
    cv2.createTrackbar(tipe_of_kernel,nameWindows,1,7,nothing)

############################################
# Dictionary of different kernels
kernels = { 1       : filter.smallBlur,
            2       : filter.largeBlur,
            3       : filter.sharpen,
            3       : filter.laplacian,
            4       : filter.edge_detect,
            5       : filter.edge_detect2,
            6       : filter.sobelX,
            7       : filter.sobelY }

while(video.isOpened()):
    #captura un frame y elimina ese frame del video
    ret, img = video.read()  #img = frame, ret = estado valido del frame si es false llego al final del archivo
    # img = cv2.resize(img,(640,360))
    if ret == False:
        break
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # imgGray = cv2.resize(imgGray,(648,360))

    ############################################
    if(1):
        # enable_filter_thershold = cv2.getTrackbarPos(Select_thres,nameWindows)
        limite = cv2.getTrackbarPos(lim_thres,nameWindows)
        area_up = cv2.getTrackbarPos(area_limit_up,nameWindows)
        area_low = cv2.getTrackbarPos(area_limit_low,nameWindows)
        enable_filter_gauss = cv2.getTrackbarPos(Select_Gauss,nameWindows)
        size_gauss = cv2.getTrackbarPos('Size Gauss',nameWindows)
        enable_filter_canny = cv2.getTrackbarPos(Select_Canny,nameWindows)
        up_canny = cv2.getTrackbarPos(canny_up,nameWindows)
        low_canny = cv2.getTrackbarPos(canny_low,nameWindows)
        enable_filter_kernel = cv2.getTrackbarPos(Select_Kernel,nameWindows)
        tipe_kernel = cv2.getTrackbarPos(tipe_of_kernel,nameWindows)
        

    
    # Pre-process Frame
    if enable_filter_gauss:
        gauss = cv2.GaussianBlur(imgGray,(size_gauss,size_gauss),0)
        imgGray = gauss
    if enable_filter_kernel:
        imgGray = cv2.filter2D(imgGray,-1,kernels[tipe_kernel])
    if enable_filter_canny:
        canny = cv2.Canny(imgGray, low_canny, up_canny)
        cv2.imshow('Canny',canny)
        imgGray = canny
        countours,_ = cv2.findContours(canny.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    else:
        _, thresh = cv2.threshold(imgGray,limite,255,cv2.THRESH_BINARY)
        cv2.imshow("Threshold", thresh)
        countours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    ############################################

    ############################################
    # Countour Analysis
    for countour in countours:
        shape,approx = sd.detect(countour)
        area= cv2.contourArea(approx)
        if  area_low<= area <=area_up:
            x,y,w,h = cv2.boundingRect(approx)
            cv2.drawContours(img,[approx],0,(255,0,0),2)
            #coordenads para escribir el texto    
            M = cv2.moments(countour) #momento de una imagen: promedio ponderado de la intencidad de pixeles 
            if M["m00"] != 0:
                #Calculo en centro de una imagen
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
            else:
                cx = approx.ravel()[0]
                cy = approx.ravel()[1]  
            # cx = approx.ravel()[0]
            # cy = approx.ravel()[1]
            cv2.putText(img,shape, (cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0))
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
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