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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


#Se establece semilal para separar datos de test del resto
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

    #Carga de datos, obtención de los valores máximos y escalado
    #Como no se hace un filtrado de atributos, pasan a ser todos features
    features_raw, target = load_wine(return_X_y=True)
    max_features = np.zeros(features_raw.shape[1])

    for i in range(0, features_raw.shape[1]):
        max_atribute = np.amax(features_raw[:, i])
        max_features[i] = max_atribute

    features = scale_to_unit(features_raw,max_features)

    # Make a train/test split using 30% test size
    # Get randomly examples to each group
    X_muestra, X_test, y_muestra, y_test = train_test_split(features, target, test_size=0.30, random_state=RANDOM_STATE)

############################################################################################
    #Definiciones de los rangos de los hiperparámetros SVM
    # r no se va a configurar ya que es el termino independiente
    #C es para todos los SVM, d es el grado del polinomio en el polinomico y gamma para todos excepto lineal
    Cs = [1000000,10000,1000,100,10,1,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001]
    degrees = [8,7,6,5,4,3,2]
    gammas = [1/13,1/12,1/11,1/10,1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2]
    #Estas 2 variables almacenan el mejor clasificador de cada tipo
    best_models_per_clasificator=[]
    best_score_per_clasificator = []
############################################################################################
    scores_iter_hiper = []  #Guarda los scores del barrido de C
    best_models = []        #Guarda los mejores modelos del barrido de C
    ##############################
    # SVM lineal OnevsOne
    ##############################
    for paramC in Cs:
        # Aplicando validacion cruzada seleccionamos el que más se aproxime a la media
        scores = []
        models = []
        #Se realiza validación cruzada separando en 5 grupos
        kf = KFold(n_splits=5)
        for train, val in kf.split(X_muestra):
            X_train, X_val, y_train, y_val = X_muestra[train], X_muestra[val], y_muestra[train], y_muestra[val]

            #Definición y entrenamiento del modelo
            model = SVC(kernel='linear',C=paramC, random_state=0)
            model_trained = OneVsOneClassifier(model).fit(X_train, y_train)
            model.fit(X_train,y_train)
            #Se guarda la precisión y el modelo obtenido
            prediction = model_trained.score(X_val, y_val)
            models.append(model_trained)
            scores.append(prediction)

        #calculo de precisión media sobre la validación cruzada
        mean = sum(scores) / float(len(scores))
        #Selección y almacenaje de la precisión más cercana a la media para extraer el modelo que está mas cercano a la media
        idx = (np.abs(scores - mean)).argmin()
        best_models.append(models[idx])
        scores_iter_hiper.append(mean)

    np_scores_iter_hiper =np.array(scores_iter_hiper)
    #Busqueda de la mayor precisión,su hiperparámetro y su modelo correspondiente
    idx = np_scores_iter_hiper.argmax()
    best_precisionOO = np.amax(np_scores_iter_hiper)
    choosen_hiperparamOO = Cs[idx]
    choosen_model = best_models[idx]
    #almacenaje del modelo que ha dado la mejor precisión para el modelo
    best_score_per_clasificator.append(best_precisionOO)
    best_models_per_clasificator.append(choosen_model)
    print("Realizando SVM lineal One vs One, el mejor hiperparámetro es",choosen_hiperparamOO,"da una precisión de ",best_precisionOO)

    ####################################################
    #SVM Lineal OnevsAll
    ###################################################

    #Resetear el vector donde se guardan los resultados
    scores_iter_hiper = []
    best_models = []

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

        idx = (np.abs(scores - mean)).argmin()
        best_models.append(models[idx])

        scores_iter_hiper.append(mean)
        #print(scores_iter_hiper)

    np_scores_iter_hiper =np.array(scores_iter_hiper)

    idx = np_scores_iter_hiper.argmax()
    best_precisionOA = np.amax(np_scores_iter_hiper)
    choosen_hiperparamOA = Cs[idx]
    choosen_model = best_models[idx]

    best_score_per_clasificator.append(best_precisionOA)
    best_models_per_clasificator.append(choosen_model)
    print("Realizando SVM lineal One vs Rest, el mejor hiperparámetro es",choosen_hiperparamOA,"da una precisión de ",best_precisionOA)

    ###########################################
    # SVM RBF Gaussiano onevsOne
    ###########################################

    #Resetear el vector donde se guardan los resultados
    accuracy_C=[]           # Guarda para cada gamma el mejor score obtenido sobre todos los C
    value_C=[]              # Guarda para cada gamma el mejor C obtenido sobre todos los C
    best_models_gamma = []  # Guarda para cada gamma el mejor modelo obtenido sobre todos los C
    for gamma in gammas:
        # Resetear el vector donde se guardan los resultados
        scores_iter_hiper = []  #Guarda los scores del barrido de C
        best_models_C = []      #Guarda los mejores modelos del barrido de C
        for paramC in Cs:
            # Aplicando validacion cruzada seleccionamos el que más se aproxime a la media
            scores = []
            models = []
            # Se realiza validación cruzada separando en 5 grupos
            kf = KFold(n_splits=5)
            for train, val in kf.split(X_muestra):
                X_train, X_val, y_train, y_val = X_muestra[train], X_muestra[val], y_muestra[train], y_muestra[val]

                # Definición y entrenamiento del modelo
                model = SVC(kernel='rbf', C=paramC,gamma=gamma, random_state=0)
                model_trained = OneVsOneClassifier(model).fit(X_train, y_train)
                # Se guarda la precisión y el modelo obtenido
                prediction = model_trained.score(X_val, y_val)
                models.append(model_trained)
                scores.append(prediction)

            # calculo de precisión media sobre la validación cruzada
            mean = sum(scores) / float(len(scores))
            # Selección y almacenaje de la precisión más cercana a la media para extraer el modelo que está mas cercano a la media
            idx = (np.abs(scores - mean)).argmin()
            best_models_C.append(models[idx])
            scores_iter_hiper.append(mean)

        np_scores_iter_hiper = np.array(scores_iter_hiper)
        # Sobre C, busqueda de la mayor precisión,su hiperparámetro y su modelo correspondiente
        idx = np_scores_iter_hiper.argmax()
        best_precisionOO = np.amax(np_scores_iter_hiper)
        choosen_hiperparamOO = Cs[idx]
        #Sobre C, almacenaje del modelo que ha dado la mejor precisión para el modelo
        best_models_gamma.append(best_models_C[idx])
        #Sobre C, almacenaje del la precisión y del valro del hiperparámetro C para cada gamma
        accuracy_C.append(best_precisionOO)
        value_C.append(choosen_hiperparamOO)

    #Busqueda de la mayor precisión
    np_accuracy_C = np.array(accuracy_C)
    idx = np_accuracy_C.argmax()
    best_precisionOO = np.max(np_accuracy_C)
    #almacenaje del modelo que ha dado la mejor precisión para el modelo
    choosen_hiperparam_C_OO=value_C[idx]
    choosen_hiperparam_Gamma_OO=gammas[idx]
    choosen_model = best_models_gamma[idx]
    #Guardado en los vectores finales
    best_score_per_clasificator.append(best_precisionOO)
    best_models_per_clasificator.append(choosen_model)

    print("Realizando SVM con kernel RBF gaussiano One vs One, el mejor hiperparámetro C es",
          choosen_hiperparam_C_OO,'el mejor hiperparámetro gamma es',choosen_hiperparam_Gamma_OO,
              "da una precisión de ", best_precisionOO)

    ###########################################
    # SVM RBF Gaussiano onevsAll
    ###########################################

    # Resetear el vector donde se guardan los resultados
    accuracy_C = []
    value_C = []
    best_models_gamma = []
    for gamma in gammas:
        # Resetear el vector donde se guardan los resultados
        scores_iter_hiper = []
        best_models_C = []
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

            idx = (np.abs(scores - mean)).argmin()
            best_models_C.append(models[idx])

            scores_iter_hiper.append(mean)
            # print(scores_iter_hiper)

        np_scores_iter_hiper = np.array(scores_iter_hiper)

        idx = np_scores_iter_hiper.argmax()
        best_precisionOA = np.amax(np_scores_iter_hiper)
        choosen_hiperparamOA = Cs[idx]

        best_models_gamma.append(best_models_C[idx])

        accuracy_C.append(best_precisionOA)
        value_C.append(choosen_hiperparamOA)

    np_accuracy_C = np.array(accuracy_C)
    idx = np_accuracy_C.argmax()
    best_precisionOA = np.max(np_accuracy_C)
    choosen_hiperparam_C_OA = value_C[idx]
    choosen_hiperparam_Gamma_OA = gammas[idx]
    choosen_model = best_models_gamma[idx]

    best_score_per_clasificator.append(best_precisionOA)
    best_models_per_clasificator.append(choosen_model)

    print("Realizando SVM con kernel RBF gaussiano One vs All, el mejor hiperparámetro C es",
          choosen_hiperparam_C_OA, 'el mejor hiperparámetro gamma es', choosen_hiperparam_Gamma_OA,
          "da una precisión de ", best_precisionOA)

    ###########################################
    # SVM Sigmoid onevsOne
    ###########################################

    # Resetear el vector donde se guardan los resultados
    accuracy_C = []  # Guarda para cada gamma el mejor score obtenido sobre todos los C
    value_C = []  # Guarda para cada gamma el mejor C obtenido sobre todos los C
    best_models_gamma = []  # Guarda para cada gamma el mejor modelo obtenido sobre todos los C
    for gamma in gammas:
        # Resetear el vector donde se guardan los resultados
        scores_iter_hiper = []  # Guarda los scores del barrido de C
        best_models_C = []  # Guarda los mejores modelos del barrido de C
        for paramC in Cs:
            # Aplicando validacion cruzada seleccionamos el que más se aproxime a la media
            scores = []
            models = []
            # Se realiza validación cruzada separando en 5 grupos
            kf = KFold(n_splits=5)
            for train, val in kf.split(X_muestra):
                X_train, X_val, y_train, y_val = X_muestra[train], X_muestra[val], y_muestra[train], y_muestra[val]

                # Definición y entrenamiento del modelo
                model = SVC(kernel='sigmoid', C=paramC, gamma=gamma, random_state=0)
                model_trained = OneVsOneClassifier(model).fit(X_train, y_train)
                # Se guarda la precisión y el modelo obtenido
                prediction = model_trained.score(X_val, y_val)
                models.append(model_trained)
                scores.append(prediction)

            # calculo de precisión media sobre la validación cruzada
            mean = sum(scores) / float(len(scores))
            # Selección y almacenaje de la precisión más cercana a la media para extraer el modelo que está mas cercano a la media
            idx = (np.abs(scores - mean)).argmin()
            best_models_C.append(models[idx])
            scores_iter_hiper.append(mean)

        np_scores_iter_hiper = np.array(scores_iter_hiper)
        # Sobre C, busqueda de la mayor precisión,su hiperparámetro y su modelo correspondiente
        idx = np_scores_iter_hiper.argmax()
        best_precisionOO = np.amax(np_scores_iter_hiper)
        choosen_hiperparamOO = Cs[idx]
        # Sobre C, almacenaje del modelo que ha dado la mejor precisión para el modelo
        best_models_gamma.append(best_models_C[idx])
        # Sobre C, almacenaje del la precisión y del valro del hiperparámetro C para cada gamma
        accuracy_C.append(best_precisionOO)
        value_C.append(choosen_hiperparamOO)

    # Busqueda de la mayor precisión
    np_accuracy_C = np.array(accuracy_C)
    idx = np_accuracy_C.argmax()
    best_precisionOO = np.max(np_accuracy_C)
    # almacenaje del modelo que ha dado la mejor precisión para el modelo
    choosen_hiperparam_C_OO = value_C[idx]
    choosen_hiperparam_Gamma_OO = gammas[idx]
    choosen_model = best_models_gamma[idx]
    # Guardado en los vectores finales
    best_score_per_clasificator.append(best_precisionOO)
    best_models_per_clasificator.append(choosen_model)

    print("Realizando SVM con kernel sigmoide One vs One, el mejor hiperparámetro C es",
          choosen_hiperparam_C_OO, 'el mejor hiperparámetro gamma es', choosen_hiperparam_Gamma_OO,
          "da una precisión de ", best_precisionOO)

    ###########################################
    # SVM Sigmoid onevsOne
    ###########################################

    # Resetear el vector donde se guardan los resultados
    accuracy_C = []
    value_C = []
    best_models_gamma = []
    for gamma in gammas:
        # Resetear el vector donde se guardan los resultados
        scores_iter_hiper = []
        best_models_C = []
        for paramC in Cs:
            # Aplicando validacion cruzada seleccionamos el que más se aproxime a la media
            scores = []
            models = []
            kf = KFold(n_splits=5)
            for train, val in kf.split(X_muestra):
                X_train, X_val, y_train, y_val = X_muestra[train], X_muestra[val], y_muestra[train], y_muestra[val]

                model = SVC(kernel='sigmoid', C=paramC, gamma=gamma, random_state=0)
                model_trained = OneVsRestClassifier(model).fit(X_train, y_train)
                prediction = model_trained.score(X_val, y_val)
                models.append(model_trained)
                scores.append(prediction)

            # print(scores)
            mean = sum(scores) / float(len(scores))

            idx = (np.abs(scores - mean)).argmin()
            best_models_C.append(models[idx])

            scores_iter_hiper.append(mean)
            # print(scores_iter_hiper)

        np_scores_iter_hiper = np.array(scores_iter_hiper)

        idx = np_scores_iter_hiper.argmax()
        best_precisionOA = np.amax(np_scores_iter_hiper)
        choosen_hiperparamOA = Cs[idx]

        best_models_gamma.append(best_models_C[idx])

        accuracy_C.append(best_precisionOA)
        value_C.append(choosen_hiperparamOA)

    np_accuracy_C = np.array(accuracy_C)
    idx = np_accuracy_C.argmax()
    best_precisionOA = np.max(np_accuracy_C)
    choosen_hiperparam_C_OA = value_C[idx]
    choosen_hiperparam_Gamma_OA = gammas[idx]
    choosen_model = best_models_gamma[idx]

    best_score_per_clasificator.append(best_precisionOA)
    best_models_per_clasificator.append(choosen_model)

    print("Realizando SVM con kernel Sigmoide One vs All, el mejor hiperparámetro C es",
          choosen_hiperparam_C_OA, 'el mejor hiperparámetro gamma es', choosen_hiperparam_Gamma_OA,
          "da una precisión de ", best_precisionOA)

    ###########################################
    # SVM Poly onevsOne
    ###########################################

    # Resetear el vector donde se guardan los resultados
    accuracy_gamma = []  # Guarda para cada gamma el mejor score obtenido sobre todos los C
    value_gamma = []
    best_models_degree = [] # Guarda para cada grado el mejor modelo obtenido sobre todos los gamma sobre todos los C
    for degree in degrees:
        accuracy_C = []  # Guarda para cada gamma el mejor score obtenido sobre todos los C
        best_models_gamma = []  # Guarda para cada gamma el mejor modelo obtenido sobre todos los C
        value_C = []  # Guarda para cada gamma el mejor C obtenido sobre todos los C
        for gamma in gammas:
            # Resetear el vector donde se guardan los resultados
            scores_iter_hiper = []  #Guarda los scores del barrido de C
            best_models_C = []      #Guarda los mejores modelos del barrido de C
            for paramC in Cs:
                # Aplicando validacion cruzada seleccionamos el que más se aproxime a la media
                scores = []
                models = []
                # Se realiza validación cruzada separando en 5 grupos
                kf = KFold(n_splits=5)
                for train, val in kf.split(X_muestra):
                    X_train, X_val, y_train, y_val = X_muestra[train], X_muestra[val], y_muestra[train], y_muestra[val]

                    # Definición y entrenamiento del modelo
                    model = SVC(kernel='poly', C=paramC,gamma=gamma,degree=degree, random_state=0)
                    model_trained = OneVsOneClassifier(model).fit(X_train, y_train)
                    # Se guarda la precisión y el modelo obtenido
                    prediction = model_trained.score(X_val, y_val)
                    models.append(model_trained)
                    scores.append(prediction)

                # calculo de precisión media sobre la validación cruzada
                mean = sum(scores) / float(len(scores))
                # Selección y almacenaje de la precisión más cercana a la media para extraer el modelo que está mas cercano a la media
                idx = (np.abs(scores - mean)).argmin()
                best_models_C.append(models[idx])
                scores_iter_hiper.append(mean)

            np_scores_iter_hiper = np.array(scores_iter_hiper)
            # Sobre C, busqueda de la mayor precisión,su hiperparámetro y su modelo correspondiente
            idx = np_scores_iter_hiper.argmax()
            best_precisionOO = np.amax(np_scores_iter_hiper)
            choosen_hiperparamOO = Cs[idx]
            #Sobre C, almacenaje del modelo que ha dado la mejor precisión para el modelo
            best_models_gamma.append(best_models_C[idx])
            #Sobre C, almacenaje del la precisión y del valro del hiperparámetro C para cada gamma
            accuracy_C.append(best_precisionOO)
            value_C.append(choosen_hiperparamOO)

        np_accuracy_C = np.array(accuracy_C)
        # Sobre C, busqueda de la mayor precisión,su hiperparámetro y su modelo correspondiente
        idx = np_accuracy_C.argmax()
        best_precision_gamma_OO = np.amax(np_accuracy_C)
        choosen_gammaOO = gammas[idx]
        # Sobre C, almacenaje del modelo que ha dado la mejor precisión para el modelo
        best_models_degree.append(best_models_gamma[idx])
        # Sobre C, almacenaje del la precisión y del valro del hiperparámetro C para cada gamma
        accuracy_gamma.append(best_precision_gamma_OO)
        value_gamma.append(choosen_gammaOO)




    #Busqueda de la mayor precisión
    np_accuracy_gamma = np.array(accuracy_gamma)
    idx = np_accuracy_gamma.argmax()
    best_precision_degree_OO = np.max(np_accuracy_gamma)
    #almacenaje del modelo que ha dado la mejor precisión para el modelo
    #choosen_hiperparam_C_OO=value_C[idx]
    #choosen_hiperparam_Gamma_OO=gammas[idx]
    choosen_model = best_models_degree[idx]

    best_score_per_clasificator.append(best_precision_degree_OO)
    best_models_per_clasificator.append(choosen_model)

    #print("Realizando SVM con kernel Sigmoide One vs All, el mejor hiperparámetro C es",
    #      choosen_hiperparam_C_OA, 'el mejor hiperparámetro gamma es', choosen_hiperparam_Gamma_OA,
    #      "da una precisión de ", best_precisionOA)

    print("Realizando SVM con kernel Polinomial One vs One, el mejor hiperparámetro C es",
          choosen_model.estimator.C, 'el mejor hiperparámetro gamma es', choosen_model.estimator.gamma,
          "y grado", choosen_model.estimator.degree,
          "da una precisión de ", best_precision_degree_OO)

    ###########################################
    # SVM Poly onevsRest
    ###########################################

    # Resetear el vector donde se guardan los resultados
    accuracy_gamma = []  # Guarda para cada gamma el mejor score obtenido sobre todos los C
    value_gamma = []
    best_models_degree = []  # Guarda para cada grado el mejor modelo obtenido sobre todos los gamma sobre todos los C
    for degree in degrees:
        accuracy_C = []  # Guarda para cada gamma el mejor score obtenido sobre todos los C
        best_models_gamma = []  # Guarda para cada gamma el mejor modelo obtenido sobre todos los C
        value_C = []  # Guarda para cada gamma el mejor C obtenido sobre todos los C
        for gamma in gammas:
            # Resetear el vector donde se guardan los resultados
            scores_iter_hiper = []  # Guarda los scores del barrido de C
            best_models_C = []  # Guarda los mejores modelos del barrido de C
            for paramC in Cs:
                # Aplicando validacion cruzada seleccionamos el que más se aproxime a la media
                scores = []
                models = []
                # Se realiza validación cruzada separando en 5 grupos
                kf = KFold(n_splits=5)
                for train, val in kf.split(X_muestra):
                    X_train, X_val, y_train, y_val = X_muestra[train], X_muestra[val], y_muestra[train], y_muestra[val]

                    # Definición y entrenamiento del modelo
                    model = SVC(kernel='poly', C=paramC, gamma=gamma, degree=degree, random_state=0)
                    model_trained = OneVsRestClassifier(model).fit(X_train, y_train)
                    # Se guarda la precisión y el modelo obtenido
                    prediction = model_trained.score(X_val, y_val)
                    models.append(model_trained)
                    scores.append(prediction)

                # calculo de precisión media sobre la validación cruzada
                mean = sum(scores) / float(len(scores))
                # Selección y almacenaje de la precisión más cercana a la media para extraer el modelo que está mas cercano a la media
                idx = (np.abs(scores - mean)).argmin()
                best_models_C.append(models[idx])
                scores_iter_hiper.append(mean)

            np_scores_iter_hiper = np.array(scores_iter_hiper)
            # Sobre C, busqueda de la mayor precisión,su hiperparámetro y su modelo correspondiente
            idx = np_scores_iter_hiper.argmax()
            best_precisionOA = np.amax(np_scores_iter_hiper)
            choosen_hiperparamOA = Cs[idx]
            # Sobre C, almacenaje del modelo que ha dado la mejor precisión para el modelo
            best_models_gamma.append(best_models_C[idx])
            # Sobre C, almacenaje del la precisión y del valro del hiperparámetro C para cada gamma
            accuracy_C.append(best_precisionOA)
            value_C.append(choosen_hiperparamOA)

        np_accuracy_C = np.array(accuracy_C)
        # Sobre C, busqueda de la mayor precisión,su hiperparámetro y su modelo correspondiente
        idx = np_accuracy_C.argmax()
        best_precision_gamma_OA = np.amax(np_accuracy_C)
        choosen_gammaOA = gammas[idx]
        # Sobre C, almacenaje del modelo que ha dado la mejor precisión para el modelo
        best_models_degree.append(best_models_gamma[idx])
        # Sobre C, almacenaje del la precisión y del valro del hiperparámetro C para cada gamma
        accuracy_gamma.append(best_precision_gamma_OA)
        value_gamma.append(choosen_gammaOA)

    # Busqueda de la mayor precisión
    np_accuracy_gamma = np.array(accuracy_gamma)
    idx = np_accuracy_gamma.argmax()
    best_precision_degree_OA = np.max(np_accuracy_gamma)
    # almacenaje del modelo que ha dado la mejor precisión para el modelo
    # choosen_hiperparam_C_OO=value_C[idx]
    # choosen_hiperparam_Gamma_OO=gammas[idx]
    choosen_model = best_models_degree[idx]

    best_score_per_clasificator.append(best_precision_degree_OA)
    best_models_per_clasificator.append(choosen_model)

    # print("Realizando SVM con kernel Sigmoide One vs All, el mejor hiperparámetro C es",
    #      choosen_hiperparam_C_OA, 'el mejor hiperparámetro gamma es', choosen_hiperparam_Gamma_OA,
    #      "da una precisión de ", best_precisionOA)

    print("Realizando SVM con kernel Polinomial One vs All, el mejor hiperparámetro C es",
          choosen_model.estimator.C, 'el mejor hiperparámetro gamma es', choosen_model.estimator.gamma,
          "y grado", choosen_model.estimator.degree,
          "da una precisión de ", best_precision_degree_OA)

    ################################################
    # Decission Tree OnevsOne
    ################################################


    max_depth_vector = np.linspace(start=1,stop=10,num=10)
    max_depth_vector=np.round(max_depth_vector)
    min_impurity_decrease_vector = np.linspace(start=0,stop=0.5,num=10)

    models = [[0 for x in range(len(max_depth_vector))] for y in range(len(min_impurity_decrease_vector))]
    scores = [[0 for x in range(len(max_depth_vector))] for y in range(len(min_impurity_decrease_vector))]


    type_classifier = "OnevsOne"
    kf = KFold(n_splits=5)

    for i in range(0, len(max_depth_vector)):
        for j in range(0, len(min_impurity_decrease_vector)):
            k = 1
            models_kfold = []
            scores_kfold = []
            #print("Para una profundidad de ", max_depth_vector[i],
            #      "y para una min_impurity de: ", min_impurity_decrease_vector[j])

            for train, val in kf.split(X_muestra):
                X_train, X_val, y_train, y_val = X_muestra[train], X_muestra[val], y_muestra[train], y_muestra[val]

                model = DecisionTreeClassifier(max_depth=max_depth_vector[i],
                                               min_impurity_decrease=min_impurity_decrease_vector[j])
                if type_classifier == "OnevsOne":
                    model_trained_k = OneVsOneClassifier(model).fit(X_train, y_train)
                else:  # "OnevsRest"
                    model_trained_k = OneVsRestClassifier(model).fit(X_train, y_train)

                prediction_k = model_trained_k.score(X_val, y_val)
                models_kfold.append(model_trained_k)
                scores_kfold.append(prediction_k)
            #    print("\tValidacion cruzada: k=", k, " score=", prediction_k)
                k += 1

            # Calculamos el modelo medio para los k-folds
            score_average = sum(scores_kfold) / float(len(scores_kfold))
            #print("\tAverage: ", score_average, "Max: ", max(scores_kfold), "min: ", min(scores_kfold))
            idx = (np.abs(np.array(scores_kfold) - score_average)).argmin()
            models[i][j] = models_kfold[idx]
            scores[i][j] = score_average

    np_scores = np.array(scores)
    max_score = np.amax(np_scores)
    optimal_depth =-1
    optimal_impurity = -1

    for i in range(0, len(max_depth_vector)):
        for j in range(0, len(min_impurity_decrease_vector)):
            if(np_scores[i,j]==max_score):
                optimal_depth = i
                optimal_impurity = j
                break

    print("Para DT con OvsO con una profundidad de ", max_depth_vector[optimal_depth], "un gini de ",
          min_impurity_decrease_vector[optimal_impurity], "y una precisión de ",
          np_scores[optimal_depth, optimal_impurity])

    best_score_per_clasificator.append(max_score)
    best_models_per_clasificator.append(models[optimal_depth][optimal_impurity])

    ################################################
    # Decission Tree OnevsRest
    ################################################


    max_depth_vector = np.linspace(start=1, stop=10, num=10)
    max_depth_vector = np.round(max_depth_vector)
    min_impurity_decrease_vector = np.linspace(start=0, stop=0.5, num=10)

    models = [[0 for x in range(len(max_depth_vector))] for y in range(len(min_impurity_decrease_vector))]
    scores = [[0 for x in range(len(max_depth_vector))] for y in range(len(min_impurity_decrease_vector))]

    type_classifier = "OnevsRest"
    kf = KFold(n_splits=5)

    for i in range(0, len(max_depth_vector)):
        for j in range(0, len(min_impurity_decrease_vector)):
            k = 1
            models_kfold = []
            scores_kfold = []
            #print("Para una profundidad de ", max_depth_vector[i],
            #      "y para una min_impurity de: ", min_impurity_decrease_vector[j])

            for train, val in kf.split(X_muestra):
                X_train, X_val, y_train, y_val = X_muestra[train], X_muestra[val], y_muestra[train], y_muestra[val]

                model = DecisionTreeClassifier(max_depth=max_depth_vector[i],
                                               min_impurity_decrease=min_impurity_decrease_vector[j])
                if type_classifier == "OnevsOne":
                    model_trained_k = OneVsOneClassifier(model).fit(X_train, y_train)
                else:  # "OnevsRest"
                    model_trained_k = OneVsRestClassifier(model).fit(X_train, y_train)

                prediction_k = model_trained_k.score(X_val, y_val)
                models_kfold.append(model_trained_k)
                scores_kfold.append(prediction_k)
            #    print("\tValidacion cruzada: k=", k, " score=", prediction_k)
            #    k += 1

            # Calculamos el modelo medio para los k-folds
            score_average = sum(scores_kfold) / float(len(scores_kfold))
            #print("\tAverage: ", score_average, "Max: ", max(scores_kfold), "min: ", min(scores_kfold))
            idx = (np.abs(np.array(scores_kfold) - score_average)).argmin()
            models[i][j] = models_kfold[idx]
            scores[i][j] = score_average

    np_scores = np.array(scores)
    max_score = np.amax(np_scores)
    optimal_depth = -1
    optimal_impurity = -1

    for i in range(0, len(max_depth_vector)):
        for j in range(0, len(min_impurity_decrease_vector)):
            if (np_scores[i, j] == max_score):
                optimal_depth = i
                optimal_impurity = j
                break

    print("Para DT con OvsA con una profundidad de ", max_depth_vector[optimal_depth],"un gini de ",min_impurity_decrease_vector[optimal_impurity], "y una precisión de ",
          np_scores[optimal_depth, optimal_impurity])

    best_score_per_clasificator.append(max_score)
    best_models_per_clasificator.append(models[optimal_depth][optimal_impurity])







    ##################################################
    # Test
    ##################################################

    np_best_score_per_clasificator = np.array(best_score_per_clasificator)

    idx = np_best_score_per_clasificator.argmax()

    final_model = best_models_per_clasificator[idx]
    print(" ")
    print("Se selecciona el modelo que mayor precisión ha generado con los datos de validación:")
    print(final_model)


    #y_estimated = final_model.predict(X_test)
    print(" ")
    print("A continuación se probarán todos los clasificadores generados para comprobar si nuestra elección es correcta:")
    print(" ")
    for i in range (0,best_models_per_clasificator.__len__()):
        if i==idx:
            score_test = final_model.score(X_test, y_test)
            print(score_test,"Precisión del que proporciona mas precisión en train")
        else:
            aux = best_models_per_clasificator[i].score(X_test,y_test)
            print(aux)



