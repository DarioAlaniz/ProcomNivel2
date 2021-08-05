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

def rescale_intensity_coustom(image):
    # nota : se debe hacer un corte (np.clip(image,(0,255))), cuando la matriz tiene numeros negativos 

    # valor para restablecer la imagen de salida de 0 a 255
    max = DeFixedInt(8,0,'U') 
    max.value = 255.0
    # Vuelvo a llevar la imagen de 0 a 255                           
    image = image * max

    print("Imagen en rango de 0 a 255".capitalize().center(espacio, "*"))
    print(image)

    # obtengo el representa en flotante para solamente tomar la parte entera
    # se obtiene el flotante representante por que por medio de la libreria fixPoint no se puede pasar 
    # directamete de un S(A,B) en U(8,0)
    imageFloat      = fixPointToFloat(image)            
    
    # Cuantifico otra vez pero ahora en formate U(8,0)
    # para solo tener en cuenta la parte entera
    imageRescale    = fixPointImage(imageFloat,8,0,'U') 

    # Obtengo el entero representante
    imageRescale    = fixPointoIntValue(imageRescale)   

    # Retorno la imagen en formate U(8,0) --> uint8 
    return imageRescale

def fixPointImage(image,NB,NBF,sMode,rMode='round',satMode='saturate'):
    # Como la libreria de fixPoint trabaja con el float de python antes de cuantificar tengo que pasarla al 
    # flota de python
    imageFlatten    = np.array(image,dtype=float)

    # Como la libreria trabaja con vectores, paso la imagen a un vector
    imageFlatten    = imageFlatten.flatten()

    # cuantifico
    imagePf         = arrayFixedInt(NB,NBF,imageFlatten,signedMode=sMode,roundMode=rMode,saturateMode=satMode)

    # reshape de la imagen cuantificada
    imagePf         = imagePf.reshape(image.shape[0],image.shape[1])

    return imagePf

def fixPointToFloat(matrix):
    # Lo trabajo como vector por simplicidad
    matrixFlatten   = matrix.flatten()

    # matris de salida definida en float(por defecto) por ser datos en float los que se van a almacenar
    imageFloat      = np.zeros(matrix.size)
    for i in range(len(matrixFlatten)):
        imageFloat[i]   = matrixFlatten[i].fValue

    # matris cuantificada, con un reshape
    imageFloat      = imageFloat.reshape(matrix.shape[0],matrix.shape[1]) 

    return imageFloat

def fixPointoIntValue(matrix):
    # Lo trabajo como vector por simplicidad
    matrixFlatten   = matrix.flatten()

    # matris de salida definida en uint8 por ser datos en U(8,0) los que se van a almacenar 
    imageIntValue   = np.zeros(matrix.size,dtype=np.uint8) 
    for i in range(len(matrixFlatten)):
        imageIntValue[i] = matrixFlatten[i].intvalue
    
    # matris cuantificada, con un reshape
    imageIntValue   = imageIntValue.reshape(matrix.shape[0],matrix.shape[1]) 

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

    # normalizo para trabajar solo con la parte fraccional y asi no aumentar muchos bit en la multiplicacion,
    # no se va de rango y solo pierdo bit en la parte menos significativa de la parte fracional
    # ej:   99*99       =   9801    cresco en la cantidad de digitos y magnitud
    #       0.99*0.99   =   0.9801  no me voy de rango y no me tengo menos bit en la parte mas significativa
    enlargedImage = enlargedImage / 255         
    #imagen cuantificada, luego del padding
    imagePf = fixPointImage(enlargedImage,nb,nbs,sig)                
    
    #Generacion de la matriz de salida, donde se guardan los resultados
    output = np.zeros(image.size,dtype=float)
    output = arrayFixedInt(nb,nbs,output,signedMode=sig,roundMode='round',saturateMode='saturate') 
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

    output = rescale_intensity_coustom(output)

    print ("Imagen despues del rescale".capitalize().center(espacio, "*"))
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



'''
# ________Notas:
# * Cuantificacion de la imagen de entrada como esta en escala de grises cada pixel es un byte por lo que es unsigned U(8,0).
#   Como se tiene distintos kernel tanto unsigned como signed hay que buscar una cuantificacion que sea adecuada para
#   trabajar tanto con unsigned como signed. Se opto por trabajar en formato S(A,A-1) para no perder el rango, ni aumentar en magnitud en la multiplicacion
#   y signado por si se quiere implementar algun kernel con valores con signo.
# * Llevar la matris de entrada en S(8,7) es desplazar los valores U(8,0) 8 bit hacia la derecha
#   y multiplicar la matris de salida por 255 seria desplazar los bit fraccionales 8 posiciones a la derecha 
# * En principio solo se va a trabajar con el gauss por eso se esta trabajando en el, si hace falta se hace las mismas pruebas con otro kernel
# ________Mejoras:
# producto de la convolucion con punto fijo por medio de la libreria.
# rescale impletentado
# cantidad de bit adecuados para el kernel en la parte fraccional : Con un S(10,9) funciona bastante bien
# ________Tareas : 
# Padding
# estimar la cantidad de bit de la parte entera y fraccionaria
# Correguir SNR con las imagenes de salida 
# 
'''
##info Peak signal to noise ratio
# cv.PSNR() 
# #https://www.ni.com/es-cr/innovations/white-papers/11/peak-signal-to-noise-ratio-as-an-image-quality-metric.html info PSNR entre 2 imagenes
# https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio forma de calculo 
# https://stackoverflow.com/questions/15495788/image-signal-to-noise-ratio-snr-and-image-quality estandares de una buena PSNR
# https://programmerclick.com/article/68551712936/ PSNR de 2 imagenes
# https://es.wikiqube.net/wiki/Signal-to-noise_ratio_(imaging) 
# https://programmerclick.com/article/9219768502/
