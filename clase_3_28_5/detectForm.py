import cv2
import matplotlib.pyplot as plt
import numpy as np

# Cargamos la imagen
original = cv2.imread("clase_3_28_5/figures/monedas.jpg")

if(1):
    cv2.imshow("original", original)


# Convertimos a escala de grises
gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

if(1):
    cv2.imshow("gris", gris)

# Aplicar suavizado Gaussiano
gauss  = cv2.GaussianBlur(gris, (5,5), 0)
#a medida que mas grande el filtro se tiene mas nitides, mejor suavisado
if(1):
    cv2.imshow("suavizado", gauss)
 
# Detectamos los bordes con Canny
canny = cv2.Canny(gauss, 50, 150)

if(1):
    cv2.imshow("canny", canny)

# Buscamos los contornos
(contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Mostramos el número de monedas por consola
print("He encontrado {} objetos".format(len(contornos)))
 
cv2.drawContours(original,contornos,-1,(0,0,255), 1)
cv2.imshow("contornos", original)
 
cv2.waitKey(0)
