import cv2
import matplotlib.pyplot as plt
import numpy as np

#en base a la relacion de aspecto identifica la imagen!!!!

# Cargamos la imagen
img     = cv2.imread("clase_4_18_6/fugPlanasSimple.jpg")
# img = cv2.imread("clase_4_18_6/articles-229261_imagen_01.png")
# img     = cv2.imread("clase_4_18_6/figurasplanas2.jpg")
#no funciona con figuras rotadas!!!!!
cv2.imshow("Image", img)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", imgGray)

if(0):
    gauss       = cv2.GaussianBlur(imgGray, (11,11), 0)
    cv2.imshow("Gauss", gauss)
    canny       = cv2.Canny(gauss, 50, 150)
    cv2.imshow("Canny", canny)
    countours,_ = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


if(0):
    # filtra la imagen en un cierto rango, en base a un cambio de color 
    # satura la imagen que esta fuera del rango
    # ayuda a no tener doble contorno en las transiciones abruptas
    _, thresh = cv2.threshold(imgGray,240,255,cv2.THRESH_BINARY)
    cv2.imshow("Threshald", thresh)
    countours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

if(1):
    gauss = cv2.GaussianBlur(imgGray, (5, 5), 0)
    thresh = cv2.threshold(gauss, 60, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Thereshald",thresh)
    countours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

for countour in countours:
    peri=cv2.arcLength(countour,True)
    approx = cv2.approxPolyDP(countour,0.01*peri,True)

    cv2.drawContours(img,[approx],0,(255,0,0),2)

    #coordenads para escribir el texto
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    print(25*'_')
    print(approx)
    # print(approx.ravel())
    M = cv2.moments(countour)
    # cX = int((M["m10"] / M["m00"]) )
    # cY = int((M["m01"] / M["m00"]) )
    # print(cX,cY)
    if x==0 and y==0:
        continue
    elif len(approx)==3:
        cv2.putText(img,"Triangle", (x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255))
    elif len(approx)==4:
        #toma los vertices de approx y mide las distancias en base a los demas vertices 
        #te da el area
        x,y,w,h = cv2.boundingRect(approx)
        aspectRatio = float(w)/h
        print(aspectRatio)
        print(x,y,w,h)
        areaRombo = (((w/2) * (w/2))/2)*4
        print(areaRombo)
        areaRombo = (w * w)/2
        print(areaRombo)
        areaContorno = cv2.contourArea(countour)
        print(areaContorno)
        print(25*'_')
        # print(aspectRatio)
        #margen de error por tener pixeles desplazados
        if aspectRatio >= 0.95 and aspectRatio <= 1.05 and (np.abs(approx.ravel()[1]-approx.ravel()[3])==0 or np.abs(approx.ravel()[1]-approx.ravel()[3])<=2):
            cv2.putText(img,"Square", (x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        else:
            if ((np.abs(approx.ravel()[0]-approx.ravel()[4])==0 or np.abs(approx.ravel()[0]-approx.ravel()[4])<=2) 
                    and (np.abs(approx.ravel()[3]-approx.ravel()[7])==0 or np.abs(approx.ravel()[3]-approx.ravel()[7])<=2)):
                cv2.putText(img,"Diamond", (x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))
            elif np.abs(approx.ravel()[0]-approx.ravel()[6])>3 :
                b1 = np.abs(approx.ravel()[0] - approx.ravel()[2])
                b2 = np.abs(approx.ravel()[4] - approx.ravel()[6])
                a1 = np.sqrt((b1-w)**2+h**2)
                a2 = np.sqrt((b2-w)**2+h**2)
                print(b1,b2,a1,a2)
                # alfa = np.arccos((np.abs(b1-w)/a1))
                # beta = np.pi-alfa
                # if beta >np.pi/2 + np.deg2rad(10) and alfa < np.pi/2 - np.deg2rad(10): #si los angulos internos son distinto de 90+-10, sera un romboide
                if np.abs(b1-b2)>=-1 and np.abs(b1-b2)<=1 and np.abs(a1-a2)>=-1 and np.abs(a1-a2)<=1:
                    cv2.putText(img,"Rhomboid", (x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))
                else:
                    cv2.putText(img,"Trapece", (x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))
            else :
                cv2.putText(img,"Rectangule", (x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))
    elif len(approx)==5:
        cv2.putText(img,"Pentagon", (x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0))
    elif len(approx)==10:
        cv2.putText(img,"Rectangule2", (x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))
    else:
        x,y,w,h = cv2.boundingRect(approx)
        relacion = float(w)/h
        if relacion>=0.95 and relacion<=1.1:
            cv2.putText(img,"Circle", (x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0))
        else :
            cv2.putText(img,"Oval", (x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0))
    cv2.imshow("Shape",img)
    cv2.waitKey(0)


# cv2.destroyWindow()
