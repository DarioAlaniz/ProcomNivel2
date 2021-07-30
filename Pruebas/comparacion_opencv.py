#https://www.sciencedirect.com/science/article/pii/S002626921000162X convolve 2d info
#https://www.sciencedirect.com/science/article/pii/S0141933104001413
import numpy as np
import cv2 as cv
from skimage.exposure.exposure import rescale_intensity,intensity_range,_output_dtype
import fileKernels as filter
import matplotlib.pyplot as plt
from tool._fixedInt import *


def rescale_intensity_coustom(image):
    #revisar!!!!
    (imin, imax) = map(float, (0 , 255))

    imin, imax = map(float, intensity_range(image))
    omin, omax = map(float, intensity_range(image, out_range,
                                            clip_negative=(imin >= 0)))
    image = np.clip(image, imin, imax)

    return image

def conv(image,kernel):
    # obtengo el largo y ancho de la imagen y el kernel
    (ih,iw)=  np.shape(image)
    (kh,kw) = np.shape(kernel)
    # saca el numero de fila y columnas para hacer un zero padding
    pad = kw//2
    # armo la imagen ampliada para tener los mismo pixeles de salida
    enlargedImage=cv.copyMakeBorder(image,pad,pad,pad,pad,cv.BORDER_REFLECT_101)
    # rotacion del kernel, opencv hace una correlacion no una convolucion por lo que no rota el kernel
    kernel = np.flip(kernel)
    # kernel=np.flipud(kernel) #roto en el eje y
    # kernel=np.fliplr(kernel) #roto en el eje x    
    output = np.zeros(np.shape(image))
    for y in np.arange(0, ih ):
        for x in np.arange(0, iw):
            # extraigo la region a para hacer el producto punto del mismo tamaño del kernel,
            # centrada en (x,y)
            reg = enlargedImage[y :y + kh, x :x + kw]
            reg1 = reg.flatten() #vector para realizar el producto 1 a 1 con el kernel
            # realizo la convolucion,
            acum=0 #suma los productos
            for k in range(kernel.size):
                acum=acum+reg1[k]*kernel.flatten()[k]
            # guardo el valor obtenido de la convolucion, 
            # en la correspondiete coordenada (x,y)
            output[y,  x] = acum

    # print(_output_dtype(output.dtype.type))
    output = rescale_intensity(output, in_range=(0,255))
    output = (output*255).astype("uint8")
    # print(_output_dtype(output.dtype.type))
    # La otra metodologia impuesta por en trabajo conseva todo el rango, pero cambia un poco en la escala ya que lleva la media
    # con histograma se aprecia este corrimiento
    # output = (((output.flatten()-min(output.flatten()))/(max(output.flatten())-min(output.flatten())))*255).astype('uint8')
    # output = output.reshape((np.shape(image)))
    return output

path = 'clase_2_21_5/actividad2/foto1.jpg'
# load the input image
imageGray = cv.imread(path,0) #flag at 0 converts directly to grayscale
# convert to gray
# imageGray = cv.cvtColor(image,cv.COLOR_BGRA2GRAY)
ih,iw = imageGray.shape
# print(ih,iw)
# resize input image in case of exceeding the limit  
if (ih > 480 and iw > 640):
    # print('cambio de resolucion')
    newImage  = cv.resize(imageGray,(640,480))
    imageGray = newImage

# dictionary of different kernels
kernels = { 'smallBlur'     : filter.smallBlur,
            'largeBlur'     : filter.largeBlur,
            'sharpen'       : filter.sharpen,
            'laplacian'     : filter.laplacian,
            'edge_detect'   : filter.edge_detect,
            'edge_detect2'  : filter.edge_detect2,
            'sobelX'        : filter.sobelX,
            'sobelY'        : filter.sobelY }

'''
# Cuantificacion de la imagen de entrada

# como esta en escala de grises cada pixel es un byte por lo que es unsigned U(8,0)
# Como se tiene distintos kernel tanto unsigned como signed hay que buscar una cuantificacion que sea adecuada para
# trabajar tanto con unsigned como signed, consulta!!!
'''
# NB=9
# NBF=0
# Mode='round'
# image_Pf = arrayFixedInt(NB,NBF,image_flatten,signedMode='S',roundMode='round',saturateMode='saturate')
# image_Pf_value = np.zeros(imageGray.size)
# for i in range(len(image_Pf)):
#     image_Pf_value[i] = image_Pf[i].fValue
# image_Pf_value = image_Pf_value.reshape(imageGray.shape[0],imageGray.shape[1]) #matris cuantificada
# print(imageGray)
# print(type(image_Pf_value))
'''
# Cuantificacion de kernel gauss
'''
if(0):
    NB=8
    NBF=6
    Mode='round'
    sizeGauss=25
    gauss = cv.getGaussianKernel(sizeGauss,0,cv.CV_32F) #filtro gauss generado para pruebas de cuantificacion
    gauss = np.array(gauss,dtype=float)
    gauss = gauss.reshape(sizeGauss//5,sizeGauss//5)
    print(gauss)
    gaussPf = arrayFixedInt(NB,NBF,gauss.flatten(),signedMode='U',roundMode='round',saturateMode='saturate')
    # print(gaussPf)
    gaussPfValue = np.zeros(25)
    for i in range(len(gaussPf)):
        gaussPfValue[i] = gaussPf[i].fValue #extraigo el representativo valor en punto en flotante del cuantificado para trabajar en python
    gaussPfValue = gaussPfValue.reshape(5,5) #kernel cuantificado
    print(gaussPfValue)
    outputCustomConvolve = conv(imageGray,gaussPfValue)
    outputOpencv = cv.filter2D(imageGray,-1,gauss)
    cv.imshow('original', imageGray)
    cv.imshow('{} coustom convolve'.format('Gauss Punto Fijo'),outputCustomConvolve)
    cv.imshow('{} opencv convolve'.format('Gauss Punto Flotante'),outputOpencv)
    # plt.show() #muestra los histogramas
    cv.waitKey(0)
    cv.destroyAllWindows()
    if(0):
        histOpenCv = cv.calcHist([outputOpencv],[0],None,[256],[0,256])
        histCustomConvolve = cv.calcHist([outputCustomConvolve],[0],None,[256],[0,256])
        plt.figure(1)
        plt.subplot(3,1,1)
        plt.hist(imageGray.flatten(),100)
        plt.subplot(3,1,2)
        plt.hist(outputOpencv.flatten(),100)
        # plt.plot(histOpenCv,color='r');plt.plot(histCustomConvolve)
        plt.subplot(3,1,3)
        plt.hist(outputCustomConvolve.flatten(),100)
        plt.show()
if(0):
    listError = []
    NB=8
    NBF=7
    Mode='round'
    sizeGauss=25
    gauss = cv.getGaussianKernel(sizeGauss,0,cv.CV_32F) #filtro gauss generado para pruebas de cuantificacion
    gauss = np.array(gauss,dtype=float)
    gauss = gauss.reshape(5,5)
    print(gauss)
    outputOpencvRef = cv.filter2D(imageGray,-1,gauss)
    for i in range(1,8):
        print(i)
        gaussPf = arrayFixedInt(NB,i,gauss.flatten(),signedMode='U',roundMode=Mode,saturateMode='saturate')
        gaussPfValue = np.zeros(25)
        for j in range(len(gaussPf)):
            gaussPfValue[j] = gaussPf[j].fValue #extraigo el representativo valor en punto en flotante del cuantificado para trabajar en python
        gaussPfValue = gaussPfValue.reshape(5,5) #kernel cuantificado
        print(gaussPfValue)
        outputCustomConvolve = conv(imageGray,gaussPfValue) 
        error = outputOpencvRef - outputCustomConvolve 
        errorSquare = error**2
        sumaErrorSquare = errorSquare.sum() 
        MSE = sumaErrorSquare / (ih*iw)
        listError.append(MSE)
    plt.figure("Media Square Error")
    plt.plot(np.arange(1,8),listError,'o-')
    plt.xlabel('NBS con NB = 8');plt.ylabel('Magnitud del error')
    plt.title("Error en la cuantificacion del kernel[gauss]")
    plt.show()
###########Gauss test#####################
# gausstest= cv.filter2D(imageGray, -1,gauss)
# gausstest1 = cv.GaussianBlur(imageGray,(5,5),0)
# cv.imshow('original', imageGray)
# cv.imshow('{} coustom convolve'.format('gauss'),gausstest)
# cv.imshow('{} opencv convolve'.format('gauss1'),gausstest1)
# cv.waitKey(0)
# cv.destroyAllWindows()

'''
# Cuantificacion de kernel edge detect 2
'''
if(0):
    NB=8
    NBF=4
    Mode='round'
    edgePf = arrayFixedInt(NB,NBF,kernels['edge_detect2'].flatten(),signedMode='S',roundMode='round',saturateMode='saturate')
    # print(edge)
    edgePfValue = np.zeros(9)
    for i in range(len(edgePfValue)):
        edgePfValue[i] = edgePf[i].fValue #extraigo el representativo valor en punto en flotante del cuantificado para trabajar en python
    edgePfValue = edgePfValue.reshape(3,3) #kernel cuantificado
    # print(gaussPfValue)
    outputCustomConvolve = conv(imageGray,edgePfValue)
    outputOpencv = cv.filter2D(imageGray,-1,kernels['edge_detect2'])
    cv.imshow('original', imageGray)
    cv.imshow('{} coustom convolve'.format('Edge detect 2 Punto Fijo'),outputCustomConvolve)
    cv.imshow('{} opencv convolve'.format('Edge detect 2 Punto Flotante'),outputOpencv)
    # plt.show() #muestra los histogramas
    cv.waitKey(0)
    cv.destroyAllWindows()
    if(0):
        histOpenCv = cv.calcHist([outputOpencv],[0],None,[256],[0,256])
        histCustomConvolve = cv.calcHist([outputCustomConvolve],[0],None,[256],[0,256])
        plt.figure(1)
        plt.subplot(3,1,1)
        plt.hist(imageGray.flatten(),100)
        plt.subplot(3,1,(2,3))
        # # plt.hist(outputOpencv.flatten(),100)
        plt.plot(histOpenCv,color='r');plt.plot(histCustomConvolve)
        # plt.subplot(3,1,3)
        # # plt.hist(outputCustomConvolve.flatten(),100)
        # # plt.plot(cdfOpencv,color='r',);plt.plot(cdfCustomConvolve)
        plt.show()
if(1):
    listError = []
    NB=8
    NBF=7
    Mode='round'
    outputOpencvRef = cv.filter2D(imageGray,-1,kernels['edge_detect2'])
    for i in range(1,8):
        print(i)
        edgePf = arrayFixedInt(NB,i,kernels['edge_detect2'].flatten(),signedMode='S',roundMode='round',saturateMode='saturate')
        # print(edge)
        edgePfValue = np.zeros(9)
        for i in range(len(edgePfValue)):
            edgePfValue[i] = edgePf[i].fValue #extraigo el representativo valor en punto en flotante del cuantificado para trabajar en python
        edgePfValue = edgePfValue.reshape(3,3) #kernel cuantificado
        print(edgePfValue)
        outputCustomConvolve = conv(imageGray,edgePfValue)
        error = outputOpencvRef - outputCustomConvolve 
        errorSquare = error**2
        sumaErrorSquare = errorSquare.sum() 
        MSE = sumaErrorSquare / (ih*iw)
        listError.append(MSE)
    plt.figure("Media Square Error")
    plt.plot(np.arange(1,8),listError,'o-')
    plt.xlabel('NBS con NB = 8');plt.ylabel('Magnitud del error')
    plt.title("Error en la cuantificacion del kernel[edge]")
    plt.show()


# cv.PSNR() 
# #https://www.ni.com/es-cr/innovations/white-papers/11/peak-signal-to-noise-ratio-as-an-image-quality-metric.html info PSNR entre 2 imagenes
# https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio forma de calculo 
# https://stackoverflow.com/questions/15495788/image-signal-to-noise-ratio-snr-and-image-quality estandares de una buena PSNR
# https://programmerclick.com/article/68551712936/ PSNR de 2 imagenes
# https://es.wikiqube.net/wiki/Signal-to-noise_ratio_(imaging) 
# https://programmerclick.com/article/9219768502/

# consultar caso donde el kernel sea no signado o signado, 
# como sacar una grafica del error si hacer, pixel a pixel como: pixelreal0/abs(pixelreal0-pixelcuant0)
# 