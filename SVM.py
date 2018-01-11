
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


RANDOM_STATE = 42


def scale_to_unit(data, max):
    '''
    Reescalado de 0-1
    :param data: datos
    :param max: array de los max para cada features
    :return: devuelve un np.array reescalado
    '''
    aux = np.array([data[:, itera] / max[itera] for itera in range(max.__len__())])
    return np.transpose(aux)


if __name__=='__main__':
    RANDOM_STATE = 42

    features_raw, target = load_wine(return_X_y=True)
    max_features = np.zeros(features_raw.shape[1])

    for i in range(0, features_raw.shape[1]):
        max_atribute = np.amax(features_raw[:, i])
        max_features[i] = max_atribute

    features = scale_to_unit(features_raw,max_features)
    #print('a')

    # Make a train/test split using 30% test size
    # Get randomly examples to each group
    X_muestra, X_test, y_muestra, y_test = train_test_split(features, target, test_size=0.20, random_state=RANDOM_STATE)

############################################################################################
    #Definiciones de los rangos de los hiperparámetros
    Cs = [1000000,10000,1000,100,10,1,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001]
    degrees = [4,3,2]
    gammas = [1/13,1/12,1/11,1/10,1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2]
    scores_iter_hiper = []
############################################################################################

    ##############################
    # SVM lineal OnevsOne
    ##############################
    for paramC in Cs:
        # Aplicando validacion cruzada seleccionamos el que más se aproxime a la media
        scores = []
        models = []
        kf = KFold(n_splits=5)
        for train, val in kf.split(X_muestra):
            X_train, X_val, y_train, y_val = X_muestra[train], X_muestra[val], y_muestra[train], y_muestra[val]

            model = SVC(kernel='linear',C=paramC, random_state=0)
            model_trained = OneVsOneClassifier(model).fit(X_train, y_train)
            prediction = model_trained.score(X_val, y_val)
            models.append(model_trained)
            scores.append(prediction)

        #print(scores)
        mean = sum(scores) / float(len(scores))

        scores_iter_hiper.append(mean)
        #print(scores_iter_hiper)

    np_scores_iter_hiper =np.array(scores_iter_hiper)

    idx = np_scores_iter_hiper.argmax()
    best_precisionOO = np.amax(np_scores_iter_hiper)
    choosen_hiperparamOO = Cs[idx]
    print("Realizando SVM lineal One vs One, el mejor hiperparámetro es",choosen_hiperparamOO,"da una precisión de ",best_precisionOO)

    ####################################################
    #SVM Lineal OnevsAll
    ###################################################

    #Resetear el vector donde se guardan los resultados
    scores_iter_hiper = []

    for paramC in Cs:
        # Aplicando validacion cruzada seleccionamos el que más se aproxime a la media
        scores = []
        models = []
        kf = KFold(n_splits=5)
        for train, val in kf.split(X_muestra):
            X_train, X_val, y_train, y_val = X_muestra[train], X_muestra[val], y_muestra[train], y_muestra[val]

            model = SVC(kernel='linear', C=paramC, random_state=0)
            model_trained = OneVsRestClassifier(model).fit(X_train, y_train)
            prediction = model_trained.score(X_val, y_val)
            models.append(model_trained)
            scores.append(prediction)

        #print(scores)
        mean = sum(scores) / float(len(scores))

        scores_iter_hiper.append(mean)
        #print(scores_iter_hiper)

    np_scores_iter_hiper =np.array(scores_iter_hiper)

    idx = np_scores_iter_hiper.argmax()
    best_precisionOA = np.amax(np_scores_iter_hiper)
    choosen_hiperparamOA = Cs[idx]
    print("Realizando SVM lineal One vs Rest, el mejor hiperparámetro es",choosen_hiperparamOA,"da una precisión de ",best_precisionOA)

    ###########################################
    # SVM RBF Gaussiano onevsOne
    ###########################################

    #Resetear el vector donde se guardan los resultados
    accuracy_C=[]
    value_C=[]
    for gamma in gammas:
        # Resetear el vector donde se guardan los resultados
        scores_iter_hiper = []
        for paramC in Cs:
            # Aplicando validacion cruzada seleccionamos el que más se aproxime a la media
            scores = []
            models = []
            kf = KFold(n_splits=5)
            for train, val in kf.split(X_muestra):
                X_train, X_val, y_train, y_val = X_muestra[train], X_muestra[val], y_muestra[train], y_muestra[val]

                model = SVC(kernel='rbf', C=paramC,gamma=gamma, random_state=0)
                model_trained = OneVsOneClassifier(model).fit(X_train, y_train)
                prediction = model_trained.score(X_val, y_val)
                models.append(model_trained)
                scores.append(prediction)

            # print(scores)
            mean = sum(scores) / float(len(scores))

            scores_iter_hiper.append(mean)
            # print(scores_iter_hiper)

        np_scores_iter_hiper = np.array(scores_iter_hiper)

        idx = np_scores_iter_hiper.argmax()
        best_precisionOO = np.amax(np_scores_iter_hiper)
        choosen_hiperparamOO = Cs[idx]

        accuracy_C.append(best_precisionOO)
        value_C.append(choosen_hiperparamOO)

    np_accuracy_C = np.array(accuracy_C)
    idx = np_accuracy_C.argmax()
    best_precisionOO = np.max(np_accuracy_C)
    choosen_hiperparam_C_OO=value_C[idx]
    choosen_hiperparam_Gamma_OO=gammas[idx]


    print("Realizando SVM con kernel RBF gaussiano One vs One, el mejor hiperparámetro C es",
          choosen_hiperparam_C_OO,'el mejor hiperparámetro gamma es',choosen_hiperparam_Gamma_OO,
              "da una precisión de ", best_precisionOO)

    ###########################################
    # SVM RBF Gaussiano onevsOne
    ###########################################

    # Resetear el vector donde se guardan los resultados
    accuracy_C = []
    value_C = []
    for gamma in gammas:
        # Resetear el vector donde se guardan los resultados
        scores_iter_hiper = []
        for paramC in Cs:
            # Aplicando validacion cruzada seleccionamos el que más se aproxime a la media
            scores = []
            models = []
            kf = KFold(n_splits=5)
            for train, val in kf.split(X_muestra):
                X_train, X_val, y_train, y_val = X_muestra[train], X_muestra[val], y_muestra[train], y_muestra[val]

                model = SVC(kernel='rbf', C=paramC, gamma=gamma, random_state=0)
                model_trained = OneVsRestClassifier(model).fit(X_train, y_train)
                prediction = model_trained.score(X_val, y_val)
                models.append(model_trained)
                scores.append(prediction)

            # print(scores)
            mean = sum(scores) / float(len(scores))

            scores_iter_hiper.append(mean)
            # print(scores_iter_hiper)

        np_scores_iter_hiper = np.array(scores_iter_hiper)

        idx = np_scores_iter_hiper.argmax()
        best_precisionOA = np.amax(np_scores_iter_hiper)
        choosen_hiperparamOA = Cs[idx]

        accuracy_C.append(best_precisionOA)
        value_C.append(choosen_hiperparamOA)

    np_accuracy_C = np.array(accuracy_C)
    idx = np_accuracy_C.argmax()
    best_precisionOO = np.max(np_accuracy_C)
    choosen_hiperparam_C_OA = value_C[idx]
    choosen_hiperparam_Gamma_OA = gammas[idx]

    print("Realizando SVM con kernel RBF gaussiano One vs All, el mejor hiperparámetro C es",
          choosen_hiperparam_C_OA, 'el mejor hiperparámetro gamma es', choosen_hiperparam_Gamma_OA,
          "da una precisión de ", best_precisionOA)

    ##################################################
    # Test
    ##################################################

