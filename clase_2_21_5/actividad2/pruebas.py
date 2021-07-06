import numpy as np
import cv2 as cv
from numpy.core.fromnumeric import shape
from skimage.exposure.exposure import rescale_intensity

kernel = np.array((
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]), dtype="int")
# k = np.arange(9)
# kernel = k.reshape((3,3))
m = np.arange(20)
image = m.reshape((5,4))

# obtengo el largo y ancho de la imagen y el kernel
(ih,iw)=np.shape(image)
(kh,kw) = np.shape(kernel)
# saca el numero de fila y columnas para hacer un zero padding
pad = kw//2
# armo la imagen ampliada para tener los mismo pixeles de salida
enlargedImage=cv.copyMakeBorder(image,pad,pad,pad,pad,cv.BORDER_CONSTANT) #cv.BORDEN_REFLEC_101 se usa en cv2.filter2D
# rotacion del kernel
kernel=np.flipud(kernel) #roto en el eje y
kernel=np.fliplr(kernel) #roto en el eje x

output = np.zeros(np.shape(image))
for y in np.arange(0, ih ):
    for x in np.arange(0, iw):
        # extraigo la region a del mismo tamaño del kernel,
        # centrada en (x,y)
        reg = enlargedImage[y :y + kh, x :x + kw]
        # realizo la convolucion
        # multiplico cada pixel con el kernel y los sumos,
        # dando el resultado coorrespondiente a
        # la coordenada (x,y) del pixel resultante
        out = (reg * kernel).sum() #este valor es un float, debe cambiar la matriz de salida a enteros
        # guardo el valor obtenido de la convolucion, 
        # en la correspondiete coordenada (x,y)
        output[y , x] = out 
        
print(output) 
# la imganen de salida al multiplicarse y sumarse se va del rango de escala de grises, por lo que se debe hacer un mapeo a escalas de grises
output = rescale_intensity(output, in_range=(0,255)) #
print(output)
# como cada pixel sigue estando en formato float se debe pasar uint8
output = (output*255).astype("uint8")
print(output)


kernels = { 'smallBlur'     : 2,
            'largeBlur'     : 3,
            'sharpen'       : 4,
            'laplacian'     : 4 }
            
print(kernels.length)        
# Primera version teniendo en cuenta el pad
# Forma teniendo en cuenta el pad
# output = np.zeros(np.shape(image))
# for y in np.arange(pad, ih+pad ):
#     for x in np.arange(pad, iw+pad):
#         # print(y,x)
#         roi = imagePadding[y - pad :y + pad +1, x-pad :x + pad+1]
#         # print(y ,y + h,x ,x + w)
#         # print(roi)
#         k = (roi * kernel).sum()
#         # print(k)
#         output[y - pad , x - pad] = k
# print(output)



