import numpy as np
import cv2 as cv
from numpy.core.fromnumeric import shape
from skimage.exposure.exposure import rescale_intensity
import argparse 
import fileKernels as filter
import matplotlib.pyplot as plt

def conv(image,kernel):
    # obtengo el largo y ancho de la imagen y el kernel
    (ih,iw)=  np.shape(image)
    (kh,kw) = np.shape(kernel)
    # saca el numero de fila y columnas para hacer un zero padding
    pad = kw//2
    # armo la imagen ampliada para tener los mismo pixeles de salida
    enlargedImage=cv.copyMakeBorder(image,pad,pad,pad,pad,cv.BORDER_REFLECT_101)
    # print(kernel)
    # rotacion del kernel, opencv hace una correlacion no una convolucion por lo que no rota el kernel
    kernel = np.flip(kernel)
    # print(kernel)
    # kernel=np.flipud(kernel) #roto en el eje y
    # kernel=np.fliplr(kernel) #roto en el eje x    
    output = np.zeros(np.shape(image))

    for y in np.arange(0, ih ):
        for x in np.arange(0, iw):
            # extraigo la region a convolucionar del mismo tamaño del kernel,
            # centrada en (x,y)
            reg = enlargedImage[y :y + kh, x :x + kw]
            # realizo la convolucion,
            # multiplico cada pixel con el kernel y los sumos,
            # dando el resultado coorrespondiente a
            # la coordenada (x,y) del pixel resultante
            out = (reg * kernel).sum() #este valor es un float, debe cambiar la matriz de salida a enteros
            # guardo el valor obtenido de la convolucion, 
            # en la correspondiete coordenada (x,y)
            output[y , x] = out       
    
    # print(output) 
    # plt.figure(1)
    # plt.subplot(3,1,1)
    # plt.hist(image.flatten(),100)
    # plt.subplot(3,1,2)
    # plt.hist(output.flatten(),100)
    
    # la imganen de salida al multiplicarse y sumarse se va del rango de escala de grises, 
    # por lo que primero normaliza y despues hace un mapeo saturando los valores negativos a cero y los mayores a 255
    output = rescale_intensity(output, in_range=(0,255))
    # como cada pixel sigue estando en formato float se debe pasar uint8, se pierde al castearlo
    output = (output*255).astype("uint8")
    
    # La otra metodologia impuesta por en trabajo conseva todo el rango, pero cambia un poco en la escala ya que lleva la media
    # con histograma se aprecia este corrimiento
    # output = (((output.flatten()-min(output.flatten()))/(max(output.flatten())-min(output.flatten())))*255).astype('uint8')
    # plt.subplot(3,1,3)
    # plt.hist(output,100)
    # output = output.reshape((np.shape(image)))
    return output

# Parse of parameter of input
ag = argparse.ArgumentParser(description="Comparation of convolution by filter2D(openCV) and convolution(coustom)")

ag.add_argument('-i','--image', help='Path to the input image', required=True)
ag.add_argument('-k','--kernel', default=20, help='Path to the kernel')

# converte argument string to object and assing them as attibutes of the namespace.
args = ag.parse_args()
# print(args)

# load the input image
imageGray = cv.imread(args.image,0) #flag at 0 converts directly to grayscale
# convert to gray
# imageGray = cv.cvtColor(image,cv.COLOR_BGRA2GRAY)

ih,iw = imageGray.shape
# print(ih,iw)
# resize input image in case of exceeding the limit  
if (ih > 480 and iw > 640):
    # print('cambio de resolucion')
    newImage  = cv.resize(imageGray,(640,480))
    imageGray = newImage
# print(imageGray.shape)

# dictionary of different kernels
kernels = { 'smallBlur'     : filter.smallBlur,
            'largeBlur'     : filter.largeBlur,
            'sharpen'       : filter.sharpen,
            'laplacian'     : filter.laplacian,
            'edge_detect'   : filter.edge_detect,
            'edge_detect2'  : filter.edge_detect2,
            'sobelX'        : filter.sobelX,
            'sobelY'        : filter.sobelY }

# plot
for key in kernels:
    outputCustomConvolve = conv(imageGray,kernels.get(key))
    outputOpencv = cv.filter2D(imageGray,-1,kernels.get(key))
    cv.imshow('original', imageGray)
    cv.imshow('{} coustom convolve'.format(key),outputCustomConvolve)
    cv.imshow('{} opencv convolve'.format(key),outputOpencv)
    # plt.show() #muestra los histogramas
    cv.waitKey(0)
    cv.destroyAllWindows()

# Pruebas con matrices simples
# k = np.arange(9)
# kernel = k.reshape((3,3))
# m = np.arange(25)
# image = m.reshape((5,5)).astype("uint8")
# a = conv(image,kernel)
# print(a)
# print(type(a))
# a = cv.filter2D(image,-1,kernel)
# print(a)
# print(type(a))
#comprobando los resultados de salida note que cv.filter2D usa BORDER_REFLECT_101 en vez de BORDER_CONSTANT como estaba usando 
# https://answers.opencv.org/question/50706/border_reflect-vs-border_reflect_101/ 



# escala grises, suavisado, y bordes
