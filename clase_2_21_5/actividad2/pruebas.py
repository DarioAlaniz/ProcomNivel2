import numpy as np
import cv2 as cv
from numpy.core.fromnumeric import shape
from skimage.exposure.exposure import rescale_intensity


kernel = np.array((
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]), dtype="int")

# k = np.arange(25)
# kernel = k.reshape((5,5))
ih,iw = kernel.shape
m = np.arange(81)
image = cv.imread("D:/dario/fulgor/2021/CV-FPGA/hardware/apps/img0.jpg",0)
npad=kernel.shape[0]//2

image=cv.copyMakeBorder(image,npad,npad,npad,npad,cv.BORDER_REFLECT_101)


print(image)

##Guardado de datos para la convolucion 
for k in range(0,image.shape[0]-(ih-1)):
    for i in range(0,image.shape[1]):
        print("\n"+10*'*')
        for j in range(0,ih):
            if(i%2==0):
                print(image[j+k][i])
            else:
                print(image[j+k][i],end=" ")
       
# obtengo el largo y ancho de la imagen y el kernel
# (ih,iw)=np.shape(image)
# (kh,kw) = np.shape(kernel)
# # saca el numero de fila y columnas para hacer un zero padding
# pad = kw//2
# # armo la imagen ampliada para tener los mismo pixeles de salida
# enlargedImage=cv.copyMakeBorder(image,pad,pad,pad,pad,cv.BORDER_REFLECT_101) #cv.BORDEN_REFLEC_101 se usa en cv2.filter2D

# paddingRow = np.zeros((pad,(iw+kw-1)))
# print(paddingRow)
# paddingCol = np.zeros(((ih),pad))
# print(paddingCol)
# imagePadding = np.column_stack((paddingCol,image))
# print(imagePadding)
# imagePadding = np.column_stack((imagePadding,paddingCol))
# print(imagePadding)
# imagePadding = np.vstack((paddingRow,imagePadding))
# print(imagePadding)
# imagePadding = np.vstack((imagePadding,paddingRow))
# print(imagePadding,imagePadding.shape)


# print(enlargedImage,enlargedImage.shape)
# # rotacion del kernel
# kernel=np.flipud(kernel) #roto en el eje y
# kernel=np.fliplr(kernel) #roto en el eje x

# output = np.zeros(np.shape(image))
# for y in np.arange(0, ih ):
#     for x in np.arange(0, iw):
#         # extraigo la region a del mismo tamaño del kernel,
#         # centrada en (x,y)
#         reg = enlargedImage[y :y + kh, x :x + kw]
#         # realizo la convolucion
#         # multiplico cada pixel con el kernel y los sumos,
#         # dando el resultado coorrespondiente a
#         # la coordenada (x,y) del pixel resultante
#         out = (reg * kernel).sum() #este valor es un float, debe cambiar la matriz de salida a enteros
#         # guardo el valor obtenido de la convolucion, 
#         # en la correspondiete coordenada (x,y)
#         output[y , x] = out 
        
# print(output) 
# # la imganen de salida al multiplicarse y sumarse se va del rango de escala de grises, por lo que se debe hacer un mapeo a escalas de grises
# output = rescale_intensity(output, in_range=(0,255)) #
# print(output)
# # como cada pixel sigue estando en formato float se debe pasar uint8
# output = (output*255).astype("uint8")
# print(output)


# kernels = { 'smallBlur'     : 2,
#             'largeBlur'     : 3,
#             'sharpen'       : 4,
#             'laplacian'     : 4 }
            
# print(kernels.length)

























