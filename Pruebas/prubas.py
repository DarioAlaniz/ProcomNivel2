from tool._fixedInt import *
import cv2 as cv
import numpy as np
gauss = cv.getGaussianKernel(25,0,cv.CV_32F)
gauss = np.array(gauss.reshape(5,5),dtype=float)
print(gauss)

fix1 = DeFixedInt(8,8,'U','round_even','saturate')
fix2 = DeFixedInt(9,8,'U','round_even','saturate')
fix3 = DeFixedInt(8,0,'U','round_even','saturate')
fix4 = DeFixedInt(8,0,'U','round_even','saturate')
fix1.value = 255.0
fix2.value = gauss[2,2]
fix3.value = 255.0
print(fix1)
print(fix2)
fixM = fix1*fix2
fixM1 = fix3*fix2
print(fixM)
print(fixM1)
print(fixM.fValue*255.0)
fix4.value = fixM1.intvalue>>8
print(fix4)









