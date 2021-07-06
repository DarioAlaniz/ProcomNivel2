## Import Packages
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

## Parser de parametros de entrada
ap = argparse.ArgumentParser( description="Convolution 2D: This function compare opencv and interative method.")

ap.add_argument("-i", "--image", required=True, help="Path to the input image")
ap.add_argument("-k", "--kernel", default=15,help="Path to the kernel")

args = ap.parse_args()

#print(args,type(args))
#print (args.image)
#print (args.kernel)

# Edge detection
edge_detect2 = np.array((
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]), dtype="int")

kernel = edge_detect2

## Carga de imagen de entrada
image = cv2.imread(args.image) 

## Convert to gray
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(gray.shape)

## Filtamos 2D
opencvOutput = cv2.filter2D(gray, -1, kernel)

## Graficamos
cv2.imshow("original", gray)
cv2.imshow("Edge Dectect - opencv", opencvOutput)
cv2.waitKey(0)
cv2.destroyAllWindows()
