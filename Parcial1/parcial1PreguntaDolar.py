from keras.utils import to_categorical
from keras import backend as K
from tensorflow.python.keras import Input
#from keras.layers import Input, Dense, SimpleRNN
#from keras.models import Model
from keras.optimizers import SGD
import numpy as np
import pandas as pd
from tensorflow.python.keras.layers import SimpleRNN, Dense
from tensorflow.python.keras.models import Model

dataset = pd.read_csv('dolar.csv',index_col='Date',parse_dates=['Date'])
print(dataset['2019':].iloc[:,[False,False,True,False,False]])
nombres = open('dolar.csv','r').read()
nombres = nombres.lower()
alfabeto = list(set(nombres))
tam_datos, tam_alfabeto = len(nombres), len(alfabeto)
#print(nombres) todo el texto
#print(tam_alfabeto,tam_datos) tamaño de cuanttos diferentes tipos de letras hay en el texto y luego tamaño de todas las letras que hay en el texto
car_a_ind = { car:ind for ind,car in enumerate(sorted(alfabeto))}
ind_a_car = { ind:car for ind,car in enumerate(sorted(alfabeto))}
print(car_a_ind)
n_a = 20    # Número de unidades en la capa oculta
entrada  = Input(shape=(None,tam_alfabeto)) #entrada nombres
a0 = Input(shape=(n_a,)) #estado oculto at-1 anterior
print(a0)
celda_recurrente = SimpleRNN(n_a, activation='tanh', return_state = True)#25 neuronas capa recurrente

capa_salida = Dense(tam_alfabeto, activation='softmax')

print(capa_salida)

hs, _ = celda_recurrente(entrada, initial_state=a0)
salida = []
salida.append(capa_salida(hs))

modelo = Model([entrada,a0],salida)
opt = SGD(lr=0.03)
modelo.compile(optimizer=opt, loss='categorical_crossentropy')

with open("texto.txt") as f:
    ejemplos = f.readlines()
ejemplos = [x.lower().strip() for x in ejemplos]
np.random.shuffle(ejemplos)
print(ejemplos)
def train_generator():
    while True:
        # Tomar un ejemplo aleatorio
        ejemplo = ejemplos[np.random.randint(0,len(ejemplos))]

        # Convertir el ejemplo a representación numérica
        X = [None] + [car_a_ind[c] for c in ejemplo]

        # Crear "Y", resultado de desplazar "X" un caracter a la derecha
        Y = X[1:] + [car_a_ind['\n']]

        # Representar "X" y "Y" en formato one-hot
        x = np.zeros((len(X),1,tam_alfabeto))
        onehot = to_categorical(X[1:],tam_alfabeto).reshape(len(X)-1,1,tam_alfabeto)
        x[1:,:,:] = onehot
        y = to_categorical(Y,tam_alfabeto).reshape(len(X),tam_alfabeto)

        # Activación inicial (matriz de ceros)
        a = np.zeros((len(X), n_a))

        yield [x, a], y

    #a = np.zeros((len(X)), n_a)

BATCH_SIZE = 10  # Número de ejemplos de entrenamiento a usar en cada iteración
NITS = 7000  # Número de iteraciones

for j in range(NITS):
    historia = modelo.fit_generator(train_generator(), steps_per_epoch=BATCH_SIZE, epochs=1, verbose=0)

        # Imprimir evolución del entrenamiento cada 1000 iteraciones
    if j % 1000 == 0:
        print('\nIteración: %d, Error: %f' % (j, historia.history['loss'][0]) + '\n')
def generar_nombre(textoUbicacion,car_a_num,tam_alfabeto,n_a):
    palabra = open(textoUbicacion,'r').readlines()
    silaba = palabra[0]
    # Inicializar x y a con ceros
    x = np.zeros((1,1,tam_alfabeto,))
    a = np.zeros((1, n_a))

    # Nombre generado y caracter de fin de linea
    nombre_generado = silaba
    fin_linea = '\n'
    car = -1

    # Iterar sobre el modelo y generar predicción hasta tanto no se alcance
    # "fin_linea" o el nombre generado llegue a los 50 caracteres
    contador = 0
    while ( contador != 4):#car != fin_linea
          # Generar predicción usando la celda RNN
          print(K.constant(x),"-----------",K.constant(a))
          a, _ = celda_recurrente(K.constant(x), initial_state=K.constant(a))
          y = capa_salida(a)
          prediccion = K.eval(y)

          # Escoger aleatoriamente un elemento de la predicción (el elemento con
          # con probabilidad más alta tendrá más opciones de ser seleccionado)
          ix = np.random.choice(list(range(tam_alfabeto)),p=prediccion.ravel())

          # Convertir el elemento seleccionado a caracter y añadirlo al nombre generado
          car = ind_a_car[ix]
          if(car == fin_linea):
              continue
          #print(car)
          nombre_generado = nombre_generado + car

          #print(nombre_generado,'-------------\n')
          # Crear x_(t+1) = y_t, y a_t = a_(t-1)
          x = to_categorical(ix,tam_alfabeto).reshape(1,1,tam_alfabeto)
          a = K.eval(a)

          # Actualizar contador y continuar
          contador += 1

          # Agregar fin de línea al nombre generado en caso de tener más de 50 caracteres
          #if (contador == 4):
            #nombre_generado += '\n'

    print(nombre_generado)
nombre = generar_nombre('silaba.txt',car_a_ind,tam_alfabeto,n_a)
palabraNew =open('silaba.txt','r').readlines()
silaba = palabraNew[0]
print(modelo.predict(3,1))