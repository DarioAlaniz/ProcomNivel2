## Importamos librerias
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib qt5
import time
from libDiagram import* #agregar en el path de la busqueda de python en vscode con pylance add path : por medio de setting
# Python › Analysis: Extra Paths

#parametros 
#consultar si estos deben ser aleatorios!!!
range_roll_off =  np.random.randint(0,11,size=10)/10.0+0.01  #np.arange(0.1,1.1,.1)
amplitud =  np.random.randint(1,6,size=10)              #np.arange(1,11,1)
oversampling = np.random.randint(2,10,size=10)           #np.arange(1,11,1)
Nbaud = np.random.randint(5,12,size=10)                 #np.arange(3,14,1)
filterRamdon = np.random.randint(0,2,size=10)           #eligo un filtro aleatorio
tipeFilter=['rc','rrc']
nomalizacion = np.random.randint(0,2,10)
Brau  = 1.0
Tbaud = 1.0/Brau
T = Tbaud

#generacion del diccionario
cases_dict = dict()
for i in range(0,10):
    cases_dict1 = { 'Test{}'.format(i+1): {'rolloff' : round(range_roll_off[i],1),
                                           'amplitud' : amplitud[i],
                                           'normalizacion' : nomalizacion[i],
                                           'oversampling' : oversampling[i],
                                           'tipo de filtro': tipeFilter[filterRamdon[i]],
                                           'numero de baudios': Nbaud[i]}
                                        }
    cases_dict.update(cases_dict1)
#interfaz
print("bienvenido".capitalize().center(50,"_"))
print("eliga algunos de siguientes test para revisar".capitalize().center(50,"_"))
for key in cases_dict:
    print(key + ': ')
    for key1 in cases_dict[key]:
        print(" "*6 + key1.capitalize() + ": " + str(cases_dict[key].get(key1)))

print('_'*60)

while(True):
    test_input = input('Elige un Test o presione <x> para salir: ')
    if test_input =='x':
        break
    elif 0<int(test_input)<11 :

        Test = 'Test{}'.format(str(test_input))
        print('Eligio '+ Test)
        for key in cases_dict[Test]:
            print(" "*6 + key.capitalize() + ": " + str(cases_dict[Test].get(key)))
        print('_'*60)
        #obtengo los parametros test elegido
        rolloff  = cases_dict[Test]['rolloff']
        amplitud = cases_dict[Test]['amplitud']
        normalizacion = cases_dict[Test]['normalizacion']
        Nbaud    = cases_dict[Test]['numero de baudios']
        oversampling = cases_dict[Test]['oversampling']
        tipo = cases_dict[Test]['tipo de filtro']

        #generacions de vectores y obtension de funciones
        t = np.arange(-0.5*Tbaud*Nbaud, 0.5*Tbaud*Nbaud, float(Tbaud+0.00001)/oversampling) #sumo un pequeño valor a Tbaud por una condicion que se va cuando el oversampling es de 4 y roll off de 0.5 da infinito la valuacion en g(t)
        g = get_filter(tipo,Tbaud,rolloff=rolloff,amplitude=amplitud,norm=normalizacion)
        g0 = g(np.array(1e-9))
        G = spec(g(t))
        f = np.linspace(-Brau/2, Brau/2, len(G))

        #PRBS
        Nsymb = 10
        b = 2*(np.random.uniform(-1,1,Nsymb)>0.0)-1
        d = np.zeros(oversampling*Nsymb)
        d[0:len(d):int(oversampling)] = b
        d=b
        d[0:10] = 2*np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])-1 #para centrar el diagrama de ojo
        #convolucion 
        t1, xt = get_signal(g,d)
        
        plt.figure()
        plt.subplot(223)
        plt.plot(t, g(t), label=r'Raised cosine $\alpha=%2.1f$'%rolloff) if tipo == 'rc' else plt.plot(t, g(t), label=r'Square root raised cosine $\alpha=%2.1f$'%rolloff)
        plt.legend()
        plt.grid()

        plt.subplot(211)
        plt.plot(f, G, label=r'Respuesta en Frecuencia $\alpha=%2.1f$'%rolloff)
        plt.legend()
        plt.grid()

        plt.subplot(224)
        drawFullEyeDiagram(xt/g0)
        plt.show()
    else :
        print('no seleccion ningun test valido'.capitalize())