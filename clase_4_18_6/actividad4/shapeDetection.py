#https://tutorial.recursospython.com/clases/
import  cv2
import numpy as np

class ShapeDetector:
    #metodo de inicializacion sin nigun parametro
    def __init__(self):
        self.name='unidentified'

    def setName(self, name):
        self.name=name

    def getName(self):
        return self.name

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
            #margen de error por tener pixeles desplazados
            if (aspectRatio >= 0.95 and aspectRatio <= 1.05 and 
                (np.abs(approx.ravel()[1]-approx.ravel()[3])==0 or np.abs(approx.ravel()[1]-approx.ravel()[3])<=2)):
                self.setName('Square')
            else:
                if ((np.abs(approx.ravel()[0]-approx.ravel()[4])==0 or np.abs(approx.ravel()[0]-approx.ravel()[4])<=2) 
                    and (np.abs(approx.ravel()[3]-approx.ravel()[7])==0 or np.abs(approx.ravel()[3]-approx.ravel()[7])<=2)):
                    self.setName("Diamond")
                elif np.abs(approx.ravel()[0]-approx.ravel()[6])>3 : #si los puntos no estan alineados puede ser un trapecio o romboide
                    b1 = np.abs(approx.ravel()[0] - approx.ravel()[2])
                    b2 = np.abs(approx.ravel()[4] - approx.ravel()[6])
                    a1 = np.sqrt((b1-w)**2+h**2)
                    a2 = np.sqrt((b2-w)**2+h**2)
                    if np.abs(b1-b2)>=-1 and np.abs(b1-b2)<=1 and np.abs(a1-a2)>=-1 and np.abs(a1-a2)<=1: #si los lados son iguales en un romboide
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
            else:
                self.setName('Oval')

        return self.getName(),approx