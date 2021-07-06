import numpy as np
import cv2 
# https://github.com/jrosebr1/imutils 
from imutils import paths  # libreria util para imagenes, solo la estoy usando para listar los paths

path = "D:/dario/fulgor/2021/clase_3_28_5/figures"

#https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def auto_canny(image,sigma=0.33):
    #obtengo la media de intencidades de los pixeles
    m       = np.median(image) 
    # construyo el valor bajo y alto del umbral en base a un porcentaje controlado por sigma
    # bajo valor de sigma indica un menor umbral
    # alto valor de sigma indica un mayor umbral
    low     = int(max(0,(1.0-sigma)*m)) 
    upper   = int(min(255,(1.0+sigma)*m))
    edged   = cv2.Canny(image,low,upper)
    return edged


for imagePath in paths.list_images(path):

    original = cv2.imread(imagePath)
    
    original  = cv2.resize(original,(400, 375))
    # Convertimos a escala de grises
    gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # ih,iw = gris.shape
    # # resize input image in case of exceeding the limit  
    # if (ih > 480 and iw > 640):
    #     # print('cambio de resolucion')
    #     newImage  = cv2.resize(gris,(640,480))
    #     gris = newImage

    # Aplicar suavizado Gaussiano
    gauss  = cv2.GaussianBlur(gris, (7,7), 0)
    #a medida que mas grande el filtro se tiene mas nitides, mejor suavisado

    # Detectamos los bordes con Canny para comprar con el automatico
    # canny = cv2.Canny(gris, 50, 150)
    cannyAuto = auto_canny(gauss,sigma=0.45)

    # Buscamos los contornos
    (contornos,_) = cv2.findContours(cannyAuto.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Mostramos el número de monedas por consola
    print("He encontrado {} objetos".format(len(contornos)))

    cv2.imshow("Original", original)
    # cv2.imshow("Gris", gris)
    # cv2.imshow('Suavizado',gauss)
    # cv2.imshow("Edge",cannyAuto)
    cv2.imshow("Procesamiento de imagen",np.hstack([gris,gauss,cannyAuto])) #concatena imagen y las muestras, deben ser todas de la misma dimension

    cv2.drawContours(original,contornos,-1,(0,0,255), 1)
    cv2.imshow("contornos", original)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

