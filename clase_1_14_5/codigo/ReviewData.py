#!/usr/bin/env python
# coding: utf-8

# In[3]:


## Primeros comandos de Python
## Comentarios de una sola linea
'''
Comentarios multilines
Comentarios
'''

## Importamos librerias
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

## Asignaciones básicas
d_num  = 5
d_flt  = 5.
d_str  = 'texto'
d_numv = [0,1,2,3];
d_strv = ['Sol','Luna','Tierra']

## Imprimir en pantalla
print ('#############################################')
print ('Este numero es: ',d_num)
print (d_str)
print ('Lista: ',d_numv,'\t','Un dato: ',d_numv[0])
print ('Lista: ',d_strv,'\t','Un dato: ',d_strv[2])
print ('El valor d_strv: %s'%d_strv[0])
print ('El valor d_strv: %2.2f'%(12.5555))
print ('Prueba Numero: {1} - Texto: {0}'.format(d_flt,d_strv[0]))
print ('Prueba Numero: {} - Texto: {}'.format(d_flt,d_strv[0]))

print ('#############################################')


# In[4]:


## Operaciones
print ('#############################################')
d_num =5
d_numv =5
suma     = 3 + d_num;    print ('Sum: ',suma)
resta    = suma - 15;    print ('Res: ',resta)
prod     = suma * resta; print ('Pro: ',prod)
div      = 5/2;          print ('Div: ',div)
div      = 5/2.;         print ('Div: ',div)  # Decimal
strv     = 'day'*2;      print (strv)         # Se repite la palabra
lists    = d_numv * 5;   print (lists)        # Se repite la lista
print ('#'*70)


# In[5]:


## Operaciones con vectores
print ('#############################################')
t_array = np.arange(0,5,2);                      print (t_array) 
t_array = np.arange(0,5)*3;                      print (t_array)
t_array = np.arange(0,5)/3;                      print (t_array)
t_array = np.arange(0,5)/3.;                     print (t_array)
t_array = np.arange(0,5)*np.arange(0,5);         print (t_array)
t_array = np.dot(np.arange(0,5),np.arange(0,5)); print (t_array)

t_matrix = np.dot(np.matrix([[0, 1, 2, 3],[0, 1, 2, 3]]),
             np.matrix([[0, 1], [2, 3],[0, 1], [2, 3]])); print (t_matrix)
print ('#############################################')


# In[6]:


## Seleccion en vectores
print ('#############################################')
t0     = np.arange(0,5);                        print (t0)
t0[0]  = 25;                                    print (t0)                          
t0[-1] = 45;                                    print (t0)                          
t0[:]  = 1;                                     print (t0)
t0[0:1]= 3;                                     print (t0)
t0[3:5]= 2;                                     print (t0)
t0[0:6]= 5;                                     print (t0)
print (len(t0),'\n',np.size(t0),'\n',np.size(t_matrix))
print (np.shape(t_matrix))
print ('#############################################')


# In[7]:


## Matrices
print ('#############################################')
t1         = np.zeros((1,5)); print (t1)
t1         = np.ones((3,5)) ; print (t1)
t1         = np.zeros((5,5)); print (t1)
t1[:,3]    = 1;               print (t1)
t1[0:5:2,:2]  = 2;            print (t1) #paso en fila de 2 y de la columna 0 a 2 
t1         = t1 + 5;          print (t1)
t1         = t1 - 5;          print (t1)
t1         = t1 * 5;          print (t1)
t1         = t1 / 5;          print (t1)
t1         = t1 * t1;         print (t1)
r,c=np.shape(t1); print (r,c,'\n',len(t1)) #shape para la dimension de la matris
print ('#############################################')


# In[8]:


## Condicional y metodo de generacion de strings
ptr = 1
if(ptr==0):
    print ("Este numero es %d"%ptr)
    print ("Hola")
elif(ptr==1):
    print ("Este numero es %d"%ptr + """\\t No-!!!#\'& %esperado""") # """ para imprimir cualquier caracter
elif(ptr>20 and ptr<=30):
    print ('>>>>>>>: ',ptr) 
else:
    print ('El valor es %d %s'%(ptr,"no esperado"))
print ('#############################################')


## Variantes de If
data = np.arange(0,10)

ptr = 16
if ptr in data:
    print ("In")
if ptr is 6:
    print ("Is")
if ptr is not 6:
    print ("Is Not")


# In[9]:


## Iterativo
t_vec = np.arange(0.,25.1,1.)
print (t_vec)
for ptr in range(len(t_vec)-1,2,-1):
    print (t_vec[ptr])
print ('#############################################')


## Ejemplo
count = 0
new_vec = []
for ptr in t_vec[0:5]:
    print (ptr),
    t_vec[count+1]=ptr+2
    print (t_vec[count])
    new_vec.append(ptr+2)
    count += 1
print ("New>> ",new_vec)
print ('#############################################')

## Iterativo
ptr = 0
o_data = 0
while(ptr < len(t_vec)):
    o_data += t_vec[ptr]
    ptr    += 1
print (o_data)
print ('#############################################')


# In[10]:


## Diccionarios
cases_dict = { 'Test1': {'mode0' : [1,2,3,4],
                         'mode1' : 'Test0',
                         'mode2' : False,
                         'mode3' : np.arange(-100,101,50)},

               'Test2': {'mode0' : [5,6,7,8],
                         'mode1' : 'Test1',
                         'mode2' : True,
                         'mode3' : np.arange(-100,101,50)},

               'Test3': {'mode0' : [9,10,11,12],
                         'mode1' : 'Test2',
                         'mode2' : False,
                         'mode3' : np.arange(-100,101,50)},
             }

print (cases_dict)
print ('*'*70)
print (dict(cases_dict['Test1']))
print ('*'*70)
print (cases_dict['Test1']['mode0'][0])

dataStr = []

for ptrStr in range(1,4):
    dataStr.append('Test{}'.format(ptrStr))
print (dataStr)
    
print ('*'*25 + 'Data Format' + '*'*25)
for ptrStr in range(1,4):
    subSet = dict(cases_dict['Test{}'.format(ptrStr)])
    print (subSet)

selData = 'Test1'
subSet  = dict(cases_dict[selData])
print (subSet['mode0'])

print ('*'*70)
selType  = 3
modeType = subSet['mode0'] if selType == 0 else subSet['mode1'] if selType == 1 else subSet['mode2'] if selType==2 else subSet['mode3']
print (modeType)


# In[11]:


## Cargar datos desde un archivo
inData = np.loadtxt('./inputBuf.out')
print (inData)

plt.figure()
plt.plot(inData[:,0],'o',label='PtrIn')
plt.plot(inData[:,1],'ro',label='Lock')
plt.plot(inData[:,2],'co',label='Enb')

plt.show()


# In[14]:


## Funcion Lambda
Fs = 100    # sampling frequency
T  = 10      # time duration we want to look at
t  = np.arange(-T, T, 1/Fs)  # the corresponding time samples

# define our two functions
x = lambda t: np.exp(-abs(t)) * (t>=0)
y = lambda t: np.sinc(t)**2

# the resulting time range, when convolving both signals
t_conv = np.arange(-2*T, 2*T, 1/Fs)[:-1]

plt.figure()
plt.subplot(121) 
plt.plot(t, x(t), label='$x(t)$')
plt.plot(t, y(t), label='$y(t)$')
z = np.convolve(x(t), y(t))/Fs
plt.plot(t_conv, z, label='$z(t)$')

# # function to calculate the spectrum of the input signal
spec = lambda x: abs(np.fft.fftshift(np.fft.fft(x, 4*len(t))))/Fs

X = spec(x(t))
Y = spec(y(t))
Z = spec(z)
f = np.linspace(-Fs/2, Fs/2, len(X))
plt.specgram(x(t))
plt.subplot(122)
plt.plot(f, X, label='$|X(f)|$')
plt.plot(f, Y, label='$|Y(f)|$')
plt.plot(f, Z, label='$|Z(f)|=|X(f)\\cdot Y(f)|$')
plt.grid()
plt.legend()
plt.show()


# In[18]:


# FOR con Listas
x = [1, 2, 3]
y = [4, 5, 6]

for i, j in zip(x, y):
    print (str(i) + " / " + str(j))


# In[23]:


## FOR Array
dataAr = np.ones((2,2))

for i,j in dataAr:
    print (i,j)


# In[24]:


## Inicializacion de funciones 
def saludar(nombre, mensaje='Hola'):
    print (mensaje, nombre) 


# In[27]:


## LLamada recursiva
def jugar(intento=1):
    respuesta = input("¿De qué color es una naranja? ")
    if respuesta != "naranja":
        if intento < 3:
            print ("\nFallaste! Inténtalo de nuevo")
            intento += 1
            jugar(intento) # Llamada recursiva
        else:
            print ("\nPerdiste!")
    else:
        print ("\nGanaste!")
jugar()


# In[31]:


## Formato
cadena = "bienvenido a mi aplicación".capitalize()
print (cadena.center(50, "="))
print (cadena.center(50, " "))


cadena = "bienvenido a mi aplicación".capitalize()
print (cadena.ljust(50, "="))

cadena = "bienvenido a mi aplicación".capitalize()
print (cadena.rjust(50, "="))

print (cadena.rjust(50, " "))

numero_factura = 1575
print (str(numero_factura).zfill(12)) 


# In[32]:


## Metodo de sustitucion
cadena = "bienvenido a mi aplicación {0}"
print (cadena.format("en Python"))

cadena = "Importe bruto: ${0} + IVA: ${1} = Importe neto: {2}"
print (cadena.format(100, 21, 121))

cadena = "Importe bruto: ${bruto} + IVA: ${iva} = Importe neto: {neto}"
print (cadena.format(bruto=100, iva=21, neto=121))

print (cadena.format(bruto=100, iva=100 * 21 / 100, neto=100 * 21 / 100 + 100))


# In[33]:


buscar = "nombre apellido"
reemplazar_por = "Juan Pérez"
print ("Estimado Sr. nombre apellido:".replace(buscar, reemplazar_por))
    
keywords = "python, guia, curso, tutorial".split(", ")
print (keywords) 


# In[ ]:




