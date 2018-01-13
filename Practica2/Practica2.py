import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

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



if __name__=="__main__":

    #######################################################################

    # 1.- Carga de datos
    full_atributos_raw = np.genfromtxt(fname='training.csv', delimiter=',', skip_header=1)
    atributos_raw = full_atributos_raw[:,1:]
    target_raw = np.genfromtxt(fname='training.csv', delimiter=',', skip_header=1, usecols=0, dtype=str )

    #######################################################################


    ##############################
    # 2.- Asignar etiquetas string a numeros
    ##############################
    labels_name = np.array(["tree", "grass", "soil", "concrete", "asphalt", "building", "car", "pool", "shadow"])
    target = np.zeros(target_raw.shape)

    for i in range(0,168):
        for label in labels_name:
            if label in target_raw[i]:
                target[i]=labels_name.tolist().index(label)

    #######################################################################

    #############################
    # 3.- Estandarizamos a la unidad
    ############################

    max_atributos = np.zeros(atributos_raw.shape[1])
    for i in range(0, atributos_raw.shape[1]):
        max_atribute = np.amax(atributos_raw[:, i])
        max_atributos[i] = max_atribute

    atributos = scale_to_unit(atributos_raw, max_atributos)


    ##############################################
    # 4.- Creamos particion train y test (20% test+80%train)
    #############################################
    RANDOM_STATE =42
    X_muestra_af, X_test_af, y_muestra_af, y_test_af = train_test_split(atributos, target, test_size=0.20, random_state=RANDOM_STATE)

    ##################################
    # Aplicamos Random Forest para extraer features de atributos
    ##################################

    clf = RandomForestClassifier(max_depth=7, random_state=0, min_impurity_decrease=0.01 )
    clf.fit(X_muestra_af, y_muestra_af)
    print("Precision para seleccion de caracteristicas con random forest",clf.score(X_test_af,y_test_af))

    aux_toprint = clf.feature_importances_
    ###################
    id_top_atributes = (-aux_toprint).argsort()[:20]
    id_top_atributes_sorted = np.sort(id_top_atributes)
    # id_top_atributes = [81, 60, 18,  61,  69,   3, 108,  22,  39,  50,  48,  27,   4, 135,  62,   6,  82,  55, 113,  8]
    # Proporciona l numero de columna (atributoi) que presenta mayor numero, con el 5, se elige el top 5
    ##############################
    print("Lista de atributos seleccioados",id_top_atributes)

    ##################################
    #Seleccionamos las columnas que resultan más relevantes
    #################################
    features = atributos[:,id_top_atributes]

    ##############################################
    # Creamos particion train y test
    #############################################
    RANDOM_STATE =42
    #X_muestra, X_test, y_muestra, y_test = train_test_split(features, target, test_size=0.20, random_state=RANDOM_STATE)




    ############################################################################
    # Clasificadores:
    # A) Random Forest
    # B) AdaBoost
    # C) Gradient Boost
    ############################################################################



    ###############################
    # A) Random Forest
    ##################################

    max_depth_vector = np.linspace(start=1, stop=30, num=30)
    max_depth_vector = np.round(max_depth_vector)
    min_impurity_decrease_vector = np.linspace(start=0, stop=0.2, num=30)

    models = [[0 for x in range(len(max_depth_vector))] for y in range(len(min_impurity_decrease_vector))]
    scores = [[0 for x in range(len(max_depth_vector))] for y in range(len(min_impurity_decrease_vector))]
    kf = KFold(n_splits=5)

    for i in range(0,len(max_depth_vector)):
        for j in range(0,len(min_impurity_decrease_vector)):
            print("\tPara un numero de estimadores de ",max_depth_vector[i],
                  "y para una min_impurity de: ",min_impurity_decrease_vector[j])

            models_kfold = []
            scores_kfold = []

            for train, val in kf.split(features):
                X_train, X_val, y_train, y_val = features[train], features[val], target[train], target[val]

                model_trained_k = RandomForestClassifier(max_depth=max_depth_vector[i],
                                                         min_impurity_decrease=min_impurity_decrease_vector[j])
                model_trained_k.fit(X_train, y_train)
                prediction_k = model_trained_k.score(X_val, y_val)
                models_kfold.append(model_trained_k)
                scores_kfold.append(prediction_k)

            # Calculamos el modelo medio para los k-folds
            score_average = sum(scores_kfold) / float(len(scores_kfold))
            score_max = max(scores_kfold)
            print("\t\tAverage: ", score_average, "Max: ", score_max, "min: ", min(scores_kfold))
            #idx = (np.abs(np.array(scores_kfold) - score_average)).argmin()
            idx = (np.abs(np.array(scores_kfold) - score_max)).argmin()
            models[i][j] = models_kfold[idx]
            #scores[i][j] = score_average
            scores[i][j] = score_max

    #choose the best model
    idmax = np.argmax(np.array(scores))
    x = int(round(idmax/len(max_depth_vector)))
    y = int(idmax-x*len(max_depth_vector))
  #  print("\n\n\n",models[x][y])
    clf1_score = scores[x][y]



    clf1 = models[x][y]


    ###############################
    # B) AdaBoost
    ##################################

    n_estimator_vector = np.linspace(start=50, stop=150, num=20)
    n_estimator_vector = np.round(n_estimator_vector)
    n_estimator_vector.astype(int)
    learning_rate_vector = np.linspace(start=0.01, stop=1, num=20)

    models = [[0 for x in range(len(n_estimator_vector))] for y in range(len(learning_rate_vector))]
    scores = [[0 for x in range(len(n_estimator_vector))] for y in range(len(learning_rate_vector))]
    kf = KFold(n_splits=5)

    for i in range(0,len(n_estimator_vector)):
        for j in range(0,len(learning_rate_vector)):
            print("\tPara un numero de estimadores de ", n_estimator_vector[i],
                  "y para un learning rate de: ", learning_rate_vector[j])

            models_kfold = []
            scores_kfold = []
            for train, val in kf.split(features):
                X_train, X_val, y_train, y_val = features[train], features[val], target[train], target[val]

                model_trained_k = AdaBoostClassifier(n_estimators=int(n_estimator_vector[i]),
                                                     learning_rate=learning_rate_vector[j])
                model_trained_k.fit(X_train, y_train)
                prediction_k = model_trained_k.score(X_val, y_val)
                models_kfold.append(model_trained_k)
                scores_kfold.append(prediction_k)

            # Calculamos el modelo medio para los k-folds
            score_average = sum(scores_kfold) / float(len(scores_kfold))
            score_max = max(scores_kfold)
            print("\t\tAverage: ", score_average, "Max: ", score_max, "min: ", min(scores_kfold),"\n")
            # idx = (np.abs(np.array(scores_kfold) - score_average)).argmin()
            idx = (np.abs(np.array(scores_kfold) - score_max)).argmin()
            models[i][j] = models_kfold[idx]
            # scores[i][j] = score_average
            scores[i][j] = score_max

    idmax = np.argmax(np.array(scores))
    x = int(round(idmax / len(n_estimator_vector)))
    y = int(idmax - x * len(n_estimator_vector))
    #print("\n\n\n", models[x][y])
    #print("Precision: ", scores[x][y])
    clf2 = models[x][y]
    clf2_score = scores[x][y]




    ####################################
    # C) Gradient Boosting
    ####################################

    n_estimator_vector = np.linspace(start=50, stop=150, num=20)
    n_estimator_vector = np.round(n_estimator_vector)
    n_estimator_vector.astype(int)
    learning_rate_vector = np.linspace(start=0.01, stop=1, num=20)

    models = [[0 for x in range(len(n_estimator_vector))] for y in range(len(learning_rate_vector))]
    scores = [[0 for x in range(len(n_estimator_vector))] for y in range(len(learning_rate_vector))]
    kf = KFold(n_splits=5)

    for i in range(0, len(n_estimator_vector)):
        for j in range(0, len(learning_rate_vector)):
            print("\tPara un numero de estimadores de ", n_estimator_vector[i],
                  "y para un learning rate de: ", learning_rate_vector[j])

            models_kfold = []
            scores_kfold = []
            for train, val in kf.split(features):
                X_train, X_val, y_train, y_val = features[train], features[val], target[train], target[val]

                model_trained_k = GradientBoostingClassifier(n_estimators=int(n_estimator_vector[i]),
                                                     learning_rate=learning_rate_vector[j])
                model_trained_k.fit(X_train, y_train)
                prediction_k = model_trained_k.score(X_val, y_val)
                models_kfold.append(model_trained_k)
                scores_kfold.append(prediction_k)

            # Calculamos el modelo medio para los k-folds
            score_average = sum(scores_kfold) / float(len(scores_kfold))
            score_max = max(scores_kfold)
            print("\t\tAverage: ", score_average, "Max: ", score_max, "min: ", min(scores_kfold), "\n")
            # idx = (np.abs(np.array(scores_kfold) - score_average)).argmin()
            idx = (np.abs(np.array(scores_kfold) - score_max)).argmin()
            models[i][j] = models_kfold[idx]
            # scores[i][j] = score_average
            scores[i][j] = score_max

    idmax = np.argmax(np.array(scores))
    x = int(round(idmax / len(n_estimator_vector)))
    y = int(idmax - x * len(n_estimator_vector))
    #print("\n\n\n", models[x][y])
   # print("Precision: ", scores[x][y])
    clf3 = models[x][y]
    clf3_score = scores[x][y]


    ########################## Imprimimos resultados
    print("\n\n\n Random Forest", clf1)
    print("Precision: ", clf1_score)
    print("\n\n\n AdaBoost", clf2)
    print("Precision: ", clf2_score)
    print("\n\n\n Gradient Boosting", clf3)
    print("Precision: ", clf3_score)


    # save the model to disk
    Autor1 = 'FelipePerez' #<- Poner aqui los nombres de los
    Autor2 = 'SergioEscobedo' #<-  dos autores.

    filename = '%s_%s.model' % (Autor1, Autor2)
    pickle.dump(clf1, open(filename, 'wb'))
