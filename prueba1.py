#TODO
'''
Normalizar los datos (usar codigo del anterior)
Iterar los hiperparámetros
Sacar curvas de interes http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
Redactar en google doc
generar matriz de confusion para el mejor de los casos antes de test y el mejor con test
Probar uno contra todos y uno contra uno

hacer DT y SVM Lineal en un principio y luego , si hay tiempo, añadir.
'''




import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

features, target = load_wine(return_X_y=True)

# Make a train/test split using 30% test size
X_muestra, X_test, y_muestra, y_test = train_test_split(features, target,
                                                    test_size=0.20,
                                                    random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_muestra, y_muestra,
                                                    test_size=0.20,
                                                    random_state=RANDOM_STATE)

print (y_test)
model = DecisionTreeClassifier(max_depth=2)
#model = LinearSVC(C=1e-6, random_state=0)
model_trained = OneVsOneClassifier(model).fit(X_train, y_train)
prediction = model_trained.score(X_val,y_val)
print(prediction)
############################################
#Cuando obtengamos la mejor precision con el de validacion, probamos el de test
############################################

#Aplicando validacion cruzada seleccionamos el que más se aproxime a la media
#DesOrdenar aleatoriamente las filas y normalizar columnas
scores=[]
models=[]
kf = KFold(n_splits=5)
for train, val in kf.split(X_muestra):
    X_train, X_val, y_train, y_val = X_muestra[train], X_muestra[val], y_muestra[train], y_muestra[val]

    model = DecisionTreeClassifier(max_depth=2)
    # model = LinearSVC(C=1e-6, random_state=0)
    model_trained = OneVsOneClassifier(model).fit(X_train, y_train)
    prediction = model_trained.score(X_val, y_val)
    models.append(model_trained)
    scores.append(prediction)

print(scores)
sum =sum(scores)/float(len(scores))

scores = np.array(scores)

idx = (np.abs(scores-sum)).argmin()
choosen_model = models[idx]
print(choosen_model)


#######################################################################