#https://tutorial.recursospython.com/clases/
import  cv2
import numpy as np

#se debe agregar en otro carpeta y crear un __init__.py dentro de la misma carpeta para poder levantar la clase desde
#otro .py
# y se puede pasar el path con la direccion a la clase donde esta su carpeta, cuando la clase se encuentre fuera de las sub carpetas
#import sys
#sys.path.append(path)
#https://www.geeksforgeeks.org/private-methods-in-python/
class ShapeDetector:
    #metodo de inicializacion sin nigun parametro
    def __init__(self):
        self.name='unidentified'

    def setName(self, name):
        self.name=name

    def getName(self):
        return self.name

    def order(self,polig):
        if (np.allclose(polig[0][0][1], polig[1][0][1],0,3)):
            return polig
        else:
            aux = [[polig[0][0]], [polig[3][0]],[polig[2][0]],[polig[1][0]]]
            aux = np.array(aux)
        return aux

    def detect(self,c):
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.01*peri,True)
        #toma los vertices de approx y mide las distancias en base a los demas vertices 
        #te da el area
        x,y,w,h = cv2.boundingRect(approx)
        # area = cv2.contourArea(c)
        if   len(approx)==3:
            self.setName('Triangle')

        elif len(approx)==4:
            aspectRatio = float(w)/h
            print(aspectRatio,x)
            #margen de error por tener pixeles desplazados
            approx=self.order(approx) #ordeno la secuencia de puntos como quiero
            x1=approx.ravel()[0];y1=approx.ravel()[1]
            x2=approx.ravel()[2];y2=approx.ravel()[3]
            x3=approx.ravel()[4];y3=approx.ravel()[5]
            x4=approx.ravel()[6];y4=approx.ravel()[7]
            if (aspectRatio >= 0.95 and aspectRatio <= 1.05 and 
                (np.abs(y1-y2)>=0 and np.abs(y1-y2)<=2)):
                self.setName('Square')
            else:
                if ((np.abs(x1-x3)>=0 and np.abs(x1-x3)<=2) and (np.abs(y2-y4)>=0 or np.abs(y2-y4)<=2)):
                    self.setName("Diamond")
                elif np.abs(x1-x4)>=2 : #si los puntos no estan alineados puede ser un trapecio o romboide
                    b1 = np.abs(x1 - x2)
                    b2 = np.abs(x3 - x4)
                    a1 = np.sqrt((b1-w)**2+h**2)
                    a2 = np.sqrt((b2-w)**2+h**2)
                    if np.abs(b1-b2)>=0 and np.abs(b1-b2)<=1 and np.abs(a1-a2)>=0 and np.abs(a1-a2)<=1: #si los lados son iguales en un romboide
                        self.setName("Rhomboid")
                    else:
                        self.setName("Trapece")
                else :
                    self.setName("Rectangule")

        elif len(approx)==5:
            self.setName('Pentagon')
        
        elif len(approx)==6:
            self.setName('Hexagon')
        elif len(approx)==10:
            self.setName('Decagon')

        else:
            relacion = float(w)/h
            if relacion>=0.95 and relacion<=1.1:
                self.setName('Circle')
            elif relacion <=0.94 or relacion >=1.2:
                self.setName('Oval')
        return self.getName(),approx