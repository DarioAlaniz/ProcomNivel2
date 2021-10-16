from tool._fixedInt import *
import cv2 as cv
import numpy as np
gauss = cv.getGaussianKernel(25,0,cv.CV_32F)
gauss = np.array(gauss.reshape(5,5),dtype=float)
matrix = np.array([ [255, 255],
                    [255, 255]], dtype=np.uint8)
print(matrix>>1)
print('*'*25)
print(gauss)

fix1 = DeFixedInt(8,8,'U','round_even','saturate')
fix2 = DeFixedInt(10,9,'U','round_even','saturate')
fix3 = DeFixedInt(8,0,'U','round_even','saturate')
fix4 = DeFixedInt(8,0,'U','round_even','saturate')

fix1.value = 255.0
fix2.value = gauss[2,2]
fix3.value = 255.0
print(fix1)
print(fix2)
print(fix3)
print('*'*25)
fixM = fix1*fix2
fixM1 = fix3*fix2
print(fixM)
print(fixM1)
print('*'*25)
print(fixM.fValue*255.0)
fix4.value = fixM1.intvalue>>9
print(fix4)







