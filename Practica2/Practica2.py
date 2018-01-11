import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pickle

def scale_to_unit(data, max):
    '''
    Reescalado de 0-1
    :param data: datos
    :param max: array de los max para cada features
    :return: devuelve un np.array reescalado
    '''
    aux = np.array([data[:, itera] / max[itera] for itera in range(max.__len__())])
    return np.transpose(aux)


###################################################################

#Carga de datos
full_atributos_raw = np.genfromtxt(fname='training.csv', delimiter=',', skip_header=1)
atributos_raw = full_atributos_raw[:,1:]
#labels = np.genfromtxt('data.txt', delimiter=',', usecols=0, dtype=str)
target_raw = np.genfromtxt(fname='training.csv', delimiter=',', skip_header=1, usecols=0, dtype=str )

###################################################################


##############################
# Asignar etiquetas string a numeros
##############################
labels_name = np.array(["tree", "grass", "soil", "concrete", "asphalt", "building", "car", "pool", "shadow"])
target = np.zeros(target_raw.shape)

for i in range(0,168):
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


##############################################
# Creamos particion train y test
#############################################
RANDOM_STATE =42
X_muestra_af, X_test_af, y_muestra_af, y_test_af = train_test_split(atributos, target, test_size=0.20, random_state=RANDOM_STATE)

###############################
# Aplicamos Random Forest para extraer features de atributos
##################################

clf = RandomForestClassifier(max_depth=7, random_state=0, min_impurity_decrease=0.01 )
clf.fit(X_muestra_af, y_muestra_af)
print("Precision para seleccion de caracteristicas con random forest",clf.score(X_test_af,y_test_af))

aux_toprint = clf.feature_importances_
###################
id_top_atributes = (-aux_toprint).argsort()[:20]
id_top_atributes_sorted = np.sort(id_top_atributes)
id_top_atributes = [81, 60, 18,  61,  69,   3, 108,  22,  39,  50,  48,  27,   4, 135,  62,   6,  82,  55, 113,  8]
# Proporciona l numero de columna (atributoi) que presenta mayor numero, con el 5, se elige el top 5
##############################
print("lista de atributos seleccioados",id_top_atributes)

##################################
#Seleccionamos las columnas que resultan más relevantes
#################################
features = atributos[:,id_top_atributes]

##############################################
# Creamos particion train y test
#############################################
RANDOM_STATE =42
X_muestra, X_test, y_muestra, y_test = train_test_split(features, target, test_size=0.20, random_state=RANDOM_STATE)

###############################
# Aplicamos Random Forest con features
##################################

clf1 = RandomForestClassifier(max_depth=10, random_state=0)
clf1.fit(X_muestra, y_muestra)
print("Precision con random forest tras seleccionar features",clf1.score(X_test,y_test))

###############################
# Aplicamos AdaBoost con features
##################################

clf2 = AdaBoostClassifier(n_estimators=50, random_state=0,learning_rate=1)
clf2.fit(X_muestra, y_muestra)
print("Precision con AdaBoost tras seleccionar features",clf2.score(X_test,y_test))


####################################
# Aplicamos Gradient Boosting
####################################

clf3 = GradientBoostingClassifier(n_estimators=150, random_state=0,learning_rate=0.01,min_samples_leaf=1)
clf3.fit(X_muestra, y_muestra)
print("Precision con Gradien Boosting tras seleccionar features",clf3.score(X_test,y_test))




# save the model to disk
Autor1 = 'FelipePerez' #<- Poner aqui los nombres de los
Autor2 = 'SergioEscobedo' #<-  dos autores.

filename = '%s_%s.model' % (Autor1, Autor2)
pickle.dump(clf1, open(filename, 'wb'))
