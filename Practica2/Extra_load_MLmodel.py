####################################################
# Cargar el modelo del clasificador aprendido      #
####################################################
test_filename  = "test.csv"
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt

model_filename = "FelipePerez_SergioEscobedo.model" #<- Poned aqui el nombre de vuestro modelo!!
modelP2 = pickle.load(open(model_filename, 'rb'))

''' ===================================================================================
De aquí en adelante debes escribir tu código para probar lo bueno que es tu modelo
Recuerda que:
 1. El fichero .csv con los datos de test estará en la misma carpeta que éste fichero.
 2. En este código debes incluir todo el prepocesado de datos que sea necesario.
    El fichero de test tiene exactamente las mismas características que el fichero de 
     entrenamiento que habeis recibido, excepto, quizas, el número de ejemplos.
 3. Para medir el rendimiento de tu modelo debes presentar por pantalla:
    -- la matriz de confusión normalizada
    -- la suma de los elementos de su diagonal principal
______________________________________________________________________________________'''


def scale_to_unit(data, max):
    '''
    Reescalado de 0-1
    :param data: datos
    :param max: array de los max para cada features
    :return: devuelve un np.array reescalado
    '''
    aux = np.array([data[:, itera] / max[itera] for itera in range(max.__len__())])
    return np.transpose(aux)

def confusion_matrix_multi(label_estimated, label_real, classes):
    matrix = np.zeros((classes, classes))
    label_estimated = label_estimated.astype(int)
    label_real = label_real.astype(int)
    for i in range(0, len(label_estimated)):
        matrix[label_real[i], label_estimated[i]] += 1

    matrix = matrix.astype('float')/matrix.sum(axis=1)[:,np.newaxis]
    diagonal = np.diagonal(matrix)
    diagonal_val = np.sum(diagonal)/classes
    return matrix,diagonal_val


def plot_confusion_matrix(cm,classes, title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

###################################################################

#Carga de datos
full_atributos_raw = np.genfromtxt(fname=test_filename, delimiter=',', skip_header=1)
atributos_raw = full_atributos_raw[:,1:]
target_raw = np.genfromtxt(fname=test_filename, delimiter=',', skip_header=1, usecols=0, dtype=str )

###################################################################


##############################
# Asignar etiquetas string a numeros
##############################
labels_name = np.array(["tree", "grass", "soil", "concrete", "asphalt", "building", "car", "pool", "shadow"])
target = np.zeros(target_raw.shape)

for i in range(0,len(target)):
    for label in labels_name:
        if label in target_raw[i]:
            target[i]=labels_name.tolist().index(label)

#######################################################################

#############################
#Estandarizamos a la unidad
############################

max_atributos = np.zeros(atributos_raw.shape[1])
for i in range(0, atributos_raw.shape[1]):
    max_atribute = np.amax(atributos_raw[:, i])
    max_atributos[i] = max_atribute

atributos = scale_to_unit(atributos_raw, max_atributos)

id_top_atributes = [81, 60, 18,  61,  69,   3, 108,  22,  39,  50,  48,  27,   4, 135,  62,   6,  82,  55, 113,  8]

features = atributos[:,id_top_atributes]

# Código para estimar las etiquetas del conjunto de test


label_estimated = modelP2.predict(features)


# Código para obtener la matriz de confusión y calculo de la suma de los elementos de la diagonal principal

confussion,diagonal = confusion_matrix_multi(label_estimated,target,9)



print("\n=== RESULTADO DE Felipe Perez y Sergio Escobedo====") #<- Poned aqui vuestros nombres!!
# Código para mostrar por pantalla los resultados

print(diagonal)
plt.figure()
plot_confusion_matrix(confussion, classes = labels_name, )
plt.show()