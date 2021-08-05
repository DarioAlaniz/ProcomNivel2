#https://www.sciencedirect.com/science/article/pii/S002626921000162X convolve 2d info
#https://www.sciencedirect.com/science/article/pii/S0141933104001413
import numpy as np
import cv2 as cv
from skimage.exposure.exposure import rescale_intensity,intensity_range,_output_dtype
import fileKernels as filter
import matplotlib.pyplot as plt
from tool._fixedInt import *

##################################
#Var globales
espacio = 100

# def rescale_intensity_coustom(image):
#     global espacio
#     #Faltar completar, simplemente esta como prueba para ver que hacer la funcion!!!
#     out_dtype = _output_dtype(image.dtype.type)
#     print ("Tipo de dato de salida de la imagen".capitalize().center(espacio, "*"))
#     print(out_dtype)
#     out_range = 'dtype'
#     imin, imax = map(float, intensity_range(image, (0,255)))
#     print ("Rango de entrada: ".capitalize().center(espacio, "*"))
#     print(imin,imax,type(imin))
#     omin, omax = map(float, intensity_range(image, out_range,
#                                             clip_negative=(imin >= 0)))
#     print ("Rango de salida: ".capitalize().center(espacio, "*"))
#     print(omin,omax,type(omin))                                                 
#     image = np.clip(image, imin, imax)
#     print ("Corte de lo valores fuera del rango de entrada".capitalize().center(espacio, "*"))
#     print(image)
#     image = (image - imin) / (imax - imin)
#     print ("Divicion por el rango maximo de entrada".capitalize().center(espacio, "*"))
#     print(image)
#     return np.asarray(image * (omax - omin) + omin, dtype=out_dtype)
    # image = image / imax

def rescale_intensity_coustom(image):
    #nota : se debe hacer un corte (np.clip(image,(0,255))), cuando la matriz tiene numeros negativos 
    max = DeFixedInt(8,0,'U') 
    max.value = 255.0                                   #valor para restablecer la imagen de salida de 0 a 255
    image = image * max
    print(image)
    imageFloat      = fixPointToFloat(image)            #obtengo el representa en flotante para solamente tomar la parte entera 
    print(imageFloat)
    imageRescale    = fixPointImage(imageFloat,8,0,'U') #imagen solo teniendo en cuenta la parte entera
    print(imageRescale)
    imageRescale    = fixPointoIntValue(imageRescale)   #Obtengo el entero representante
    return imageRescale

def fixPointImage(image,NB,NBF,sMode,rMode='round',satMode='saturate'):
    imageFlatten    = np.array(image,dtype=float)
    imageFlatten    = imageFlatten.flatten()
    imagePf         = arrayFixedInt(NB,NBF,imageFlatten,signedMode=sMode,roundMode=rMode,saturateMode=satMode)
    imagePf         = imagePf.reshape(image.shape[0],image.shape[1])
    return imagePf

def fixPointToFloat(matrix):
    matrixFlatten   = matrix.flatten()
    imageFloat      = np.zeros(matrix.size)
    for i in range(len(matrixFlatten)):
        imageFloat[i]   = matrixFlatten[i].fValue
    imageFloat      = imageFloat.reshape(matrix.shape[0],matrix.shape[1]) #matris cuantificada
    return imageFloat

def fixPointoIntValue(matrix):
    matrixFlatten   = matrix.flatten()
    imageIntValue   = np.zeros(matrix.size,dtype=np.uint8) #definida en uint8 por guardar datos en 
    for i in range(len(matrixFlatten)):
        imageIntValue[i] = matrixFlatten[i].intvalue
    imageIntValue   = imageIntValue.reshape(matrix.shape[0],matrix.shape[1]) #matris cuantificada
    return imageIntValue

def conv(image,kernel):
    global espacio
    nb=8
    nbs=7
    sig = 'S'
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

    #normalizo para usar un notacion Q(7) no tengo en cuenta el signo
    enlargedImage = enlargedImage / 255         

    #imagen cuantificada, luego del padding
    imagePf = fixPointImage(enlargedImage,nb,nbs,sig)                
    
    #Generacion de la matriz de salida
    output = np.zeros(image.size,dtype=float)
    output = arrayFixedInt(nb,nbs,output,signedMode=sig,roundMode='round',saturateMode='saturate') #matris de salida donde se guardan los resultados
    output = output.reshape(np.shape(image))
    # print(output)

    #Convolucion 2D
    for y in np.arange(0, ih ):
        for x in np.arange(0, iw):
            # extraigo la region a para hacer el producto punto del mismo tamaño del kernel,
            # centrada en (x,y)
            reg = imagePf[y :y + kh, x :x + kw]

            #vector para realizar el producto 1 a 1 con el kernel
            reg1 = reg.flatten() 
            
            acum=DeFixedInt(nb,nbs,signedMode=sig) #almacena los productos
            for k in range(kernel.size):
                acum=acum+reg1[k]*kernel.flatten()[k]
            # guardo el valor obtenido de la convolucion, 
            # en la correspondiete coordenada (x,y)
            output[y,  x] = acum

            # Con punto flotante
            # output [y,x] = (reg * kernel).sum()
    
    
    print ("Imagen antes del rescale:".capitalize().center(espacio, "*"))
    print(output)
    # output = rescale_intensity(output, in_range=(0,255))
    output = rescale_intensity_coustom(output)
    print ("Imagen despues del rescale".capitalize().center(espacio, "*"))
    print(output)
    output = (output*255).astype("uint8") #para poder usarla en la openCV
    print ("Vuelve a multiplicar por 255 y toma la parte entera".capitalize().center(espacio, "*"))
    print(output)
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
# ________Notas:
# en principio solo se va a trabajar con el gauss por eso se esta trabajando en el, si hace falta se hace las mismas pruebas con otro kernel
# ________Mejoras:
# producto de la convolucion con punto fijo por medio de la libreria.
# ________Tareas : 
# Ver rescale, estimando el tamaño de final de cada pixel en punto fijo para poder realizarlo 
# Ver padding
# estimar la cantidad de bit de la parte entera y fraccionaria
# Ver el snr del kernel gauss para asignar la cantidad de bit adecuados para la parte fraccional
# 
'''

'''
# Cuantificacion de kernel gauss
'''
if(0):
    SNRVect = []
    PSNRVect= []
    NB=16
    NBF=15
    Mode='round'
    sizeGauss=25
    gauss = cv.getGaussianKernel(sizeGauss,0,cv.CV_32F) #filtro gauss generado para pruebas de cuantificacion
    gauss = np.array(gauss,dtype=float)
    gauss = gauss.reshape(5,5)
    # print(gauss)
    listErrorImage=[]
    outputOpencvRef = cv.filter2D(imageGray,-1,gauss)
    signal  = np.dot(outputOpencvRef.flatten(),outputOpencvRef.flatten()) #para el calculo del SNR
    for i in range(1,NBF+1):
        print('*'*25)
        print('Numero de bit fraccionales: ',i)
        gaussPf = fixPointImage(gauss,NB,i,'U','round','saturate')
        # print(gaussPfValue)
        # errorGauss =  gauss - gaussPfValue
        # SNR =10*np.log10(np.dot(gauss.flatten(),gauss.flatten())/np.dot(errorGauss.flatten(),errorGauss.flatten()))
        ###Verificar el SNR!!!
        outputCustomConvolve    = conv(imageGray,gaussPf)
        error   = outputOpencvRef - outputCustomConvolve
        noise   = np.dot(error.flatten(),error.flatten())
        print(signal)
        print(noise)
        SNR     = 10*np.log10(signal/noise)
        print(SNR)
        #################
        listErrorImage.append(error.sum()/(iw*ih)) #erro cuadratico medio
        PSNRVect.append(cv.PSNR(outputOpencvRef,outputCustomConvolve))
        SNRVect.append(SNR)
    print(len(listErrorImage))
    # cv.imshow("Error ",np.hstack(listErrorImage[:3]))
    # cv.imshow("Error 1 ",np.hstack(listErrorImage[3:]))
    cv.waitKey(0)
    cv.destroyAllWindows()
    plt.figure(1)
    plt.subplot(311)
    plt.plot(np.arange(1,NBF+1),SNRVect,'o-')
    # plt.xlabel('NBS con NB = {}'.format(NB));plt.ylabel('Magnitud[dB]')
    plt.title("SNR[Signal to noise ratio]")
    # Pruebas con MSE y PSNR para ver si el error entre imagenes disminuye 
    plt.subplot(312)
    plt.plot(np.arange(1,NBF+1),listErrorImage,'o-')
    # plt.xlabel('NBS con NB = {}'.format(NB));plt.ylabel('Magnitud')
    plt.title("MSE[median square error]")
    plt.subplot(313)
    plt.plot(np.arange(1,NBF+1),PSNRVect,'o-')
    plt.xlabel('NBS con NB = {}'.format(NB));plt.ylabel('Magnitud[dB]')
    plt.title("PSNR[peak signal to noise ratio]")
    plt.show()
'''

Prueba para ver la imagen de salida con el kernel cuantificado
''' 
if(1):
    NB=10
    NBF=9
    sizeGauss=25
    gauss = cv.getGaussianKernel(sizeGauss,0,cv.CV_32F) #filtro gauss generado para pruebas de cuantificacion
    gauss = gauss.reshape(sizeGauss//5,sizeGauss//5)
    gaussPf = fixPointImage(gauss,NB,NBF,'S','round','saturate') #convierto a formato Q(9)
    print ("kernel cuantificado".capitalize().center(espacio, "*"))
    print(gaussPf)
    outputCustomConvolve = conv(imageGray,gaussPf)
    outputOpencv = cv.filter2D(imageGray,-1,gauss)
    cv.imshow('original', imageGray)
    cv.imshow('{} coustom convolve'.format('Gauss Punto Fijo'),outputCustomConvolve)
    cv.imshow('{} opencv convolve'.format('Gauss Punto Flotante'),outputOpencv)
    # cv.imshow('Error',error)
    # plt.show() #muestra los histogramas
    cv.waitKey(0)
    cv.destroyAllWindows()
    if(0):
        # histOpenCv = cv.calcHist([outputOpencv],[0],None,[256],[0,256])
        # histCustomConvolve = cv.calcHist([outputCustomConvolve],[0],None,[256],[0,256])
        plt.figure(1)
        # plt.subplot(3,1,1)
        # plt.hist(error.flatten(),100)
        plt.subplot(3,1,1)
        plt.hist(outputOpencv.flatten(),100)
        # plt.plot(histOpenCv,color='r');plt.plot(histCustomConvolve)
        plt.subplot(3,1,2)
        plt.hist(outputCustomConvolve.flatten(),100)
        plt.show()
###########Gauss test#####################
# gausstest= cv.filter2D(imageGray, -1,gauss)
# gausstest1 = cv.GaussianBlur(imageGray,(5,5),0)
# cv.imshow('original', imageGray)
# cv.imshow('{} coustom convolve'.format('gauss'),gausstest)
# cv.imshow('{} opencv convolve'.format('gauss1'),gausstest1)
# cv.waitKey(0)
# cv.destroyAllWindows()


##info Peak signal to noise ratio
# cv.PSNR() 
# #https://www.ni.com/es-cr/innovations/white-papers/11/peak-signal-to-noise-ratio-as-an-image-quality-metric.html info PSNR entre 2 imagenes
# https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio forma de calculo 
# https://stackoverflow.com/questions/15495788/image-signal-to-noise-ratio-snr-and-image-quality estandares de una buena PSNR
# https://programmerclick.com/article/68551712936/ PSNR de 2 imagenes
# https://es.wikiqube.net/wiki/Signal-to-noise_ratio_(imaging) 
# https://programmerclick.com/article/9219768502/
