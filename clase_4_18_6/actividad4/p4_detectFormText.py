import cv2
import matplotlib.pyplot as plt
import numpy as np
from shapeDetection import ShapeDetector

#en base a la relacion de aspecto identifica la imagen!!!!

# Cargamos la imagen
#original = cv2.imread("figures/monedas.jpg")
# img     = cv2.imread("clase_4_18_6/fugPlanasSimple.jpg")
# img     = cv2.imread("clase_4_18_6/rectangulo.png")
img       = cv2.imread("clase_4_18_6/cuadrado_simple.jpg")
# img     = cv2.imread("clase_4_18_6/figurasplanas2.jpg")
# img = cv2.imread("clase_4_18_6/articles-229261_imagen_01.png")
# img = cv2.imread("info/shape-detection/shapes_and_colors.png") #no funciona con figuras rotadas, corregir

cv2.imshow("Image", img)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", imgGray)
sd = ShapeDetector()

if(0):
    gauss       = cv2.GaussianBlur(imgGray, (11,11), 0)
    cv2.imshow("Gauss", gauss)
    canny       = cv2.Canny(gauss, 50, 150)
    cv2.imshow("Canny", canny)
    countours,_ = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

if(1):
    # filtra la imagen en un cierto rango, en base a un cambio de color 
    # satura la imagen que esta fuera del rango
    # ayuda a no tener doble contorno en las transiciones abruptas
    # _, thresh = cv2.threshold(imgGray,240,255,cv2.THRESH_BINARY)
    _, thresh = cv2.threshold(imgGray,202,255,cv2.THRESH_BINARY)
    cv2.imshow("Threshald", thresh)
    countours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
if(0):
    gauss = cv2.GaussianBlur(imgGray, (5, 5), 0)
    thresh = cv2.threshold(gauss, 60, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Thereshald",thresh)
    countours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

for countour in countours:
    shape,approx = sd.detect(countour)
    print(approx)
    x,y,w,h = cv2.boundingRect(approx)
    if approx.ravel()[0] == 0 and approx.ravel()[1] == 00 : #no toma el contorno de la imagen
        continue
    else:
        cv2.drawContours(img,[approx],0,(255,0,0),2)
        #coordenads para escribir el texto
        #https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
        #http://datahacker.rs/006-opencv-projects-how-to-detect-contours-and-match-shapes-in-an-image-in-python/
        
        M = cv2.moments(countour) #momento de una imagen: promedio ponderado de la intencidad de pixeles 
        #Calculo en centro de una imagen
        # cx = int(M["m10"]/M["m00"])
        # cy = int(M["m01"]/M["m00"])
        cx = approx.ravel()[0]
        cy = approx.ravel()[1]
        cv2.putText(img,shape, (cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0))
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("Shape",img)
        

cv2.waitKey(0)
# cv2.destroyAllWindows()