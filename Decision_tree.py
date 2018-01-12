import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

def scale_to_unit(data):
    '''
    Reescalado de 0-1
    :param data: datos
    :param max: array de los max para cada features
    :return: devuelve un np.array reescalado
    '''
    max_features = np.zeros(data.shape[1])
    for i in range(0, data.shape[1]):
        max_atribute = np.amax(data[:, i])
        max_features[i] = max_atribute

    aux = np.array([data[:, itera] / max_features[itera] for itera in range(len(max_features))])
    return np.transpose(aux)

if __name__== "__main__":

    RANDOM_STATE = 42
    features, target = load_wine(return_X_y=True)

    # Make a train/test split using 30% test size
    X_muestra, X_test, y_muestra, y_test = train_test_split(features, target,
                                                        test_size=0.20,
                                                        random_state=RANDOM_STATE)
    ############ Normalizamos la muestra/test
    X_muestra = scale_to_unit(X_muestra)
    X_test = scale_to_unit(X_test)




    X_train, X_val, y_train, y_val = train_test_split(X_muestra, y_muestra,
                                                        test_size=0.20,
                                                    random_state=RANDOM_STATE)

    type_classifier = "OnevsRest"
    print("Has seleccionado: ",type_classifier)



    max_depth_vector = np.linspace(start=1,stop=10,num=10)
    max_depth_vector=np.round(max_depth_vector)
    min_impurity_decrease_vector = np.linspace(start=0,stop=0.5,num=10)

    models = [[0 for x in range(len(max_depth_vector))] for y in range(len(min_impurity_decrease_vector))]
    scores = [[0 for x in range(len(max_depth_vector))] for y in range(len(min_impurity_decrease_vector))]



    for i in range(0,len(max_depth_vector)):
        for j in range(0,len(min_impurity_decrease_vector)):
            print("Para una profundidad de ",max_depth_vector[i],
                  "y para una min_impurity de: ",min_impurity_decrease_vector[j])
            model = DecisionTreeClassifier(max_depth=max_depth_vector[i],min_impurity_decrease=min_impurity_decrease_vector[j])

            if type_classifier == "OnevsOne":
                model_trained = OneVsOneClassifier(model).fit(X_train, y_train)
            else: #"OnevsRest"
                model_trained = OneVsRestClassifier(model).fit(X_train, y_train)

            prediction = model_trained.score(X_val, y_val)
            scores[i][j]=prediction
            models[i][j]=model_trained

    f, axarr = plt.subplots(2, 5)
    if type_classifier == "OnevsOne":
        plt.suptitle("OneVsOneClassifier: Decision Tree")
    else:
        plt.suptitle("OneVsRestClassifier: Decision Tree")

    for i in range(2):
        for j in range(5):
            title = "Depth: "+str(max_depth_vector[(i+1)*(j+1)-1])
            axarr[i, j].plot(min_impurity_decrease_vector,scores[i*j][:])
            axarr[i, j].plot(min_impurity_decrease_vector,scores[i*j][:],'x')

            axarr[i, j].set_title(title)
            axarr[i, j].grid()
            #axarr[i,j].set_ylabel("Score")

    plt.show()


    ##########################################
    #   Validacion cruzada
    ##########################################

    type_classifier = "OnevsRest"
    kf = KFold(n_splits=5)

    for i in range(0, len(max_depth_vector)):
        for j in range(0, len(min_impurity_decrease_vector)):
            k=1
            models_kfold = []
            scores_kfold = []
            print("Para una profundidad de ", max_depth_vector[i],
                  "y para una min_impurity de: ", min_impurity_decrease_vector[j])

            for train, val in kf.split(X_muestra):
                X_train, X_val, y_train, y_val = X_muestra[train], X_muestra[val], y_muestra[train], y_muestra[val]

                model = DecisionTreeClassifier(max_depth=max_depth_vector[i],min_impurity_decrease=min_impurity_decrease_vector[j])
                if type_classifier == "OnevsOne":
                    model_trained_k = OneVsOneClassifier(model).fit(X_train, y_train)
                else:  # "OnevsRest"
                    model_trained_k = OneVsRestClassifier(model).fit(X_train, y_train)

                prediction_k = model_trained_k.score(X_val, y_val)
                models_kfold.append(model_trained_k)
                scores_kfold.append(prediction_k)
                print("\tValidacion cruzada: k=",k," score=",prediction_k)
                k += 1

            #Calculamos el modelo medio para los k-folds
            score_average = sum(scores_kfold) / float(len(scores_kfold))
            print("\tAverage: ",score_average,"Max: ",max(scores_kfold),"min: ",min(scores_kfold))
            idx = (np.abs(np.array(scores_kfold) - score_average)).argmin()
            models[i][j] = models_kfold[idx]
            scores[i][j] = score_average

    f, axarr = plt.subplots(2, 5)
    if type_classifier == "OnevsOne":
        plt.suptitle("Cross Validation:OneVsOneClassifier: Decision Tree")
    else:
        plt.suptitle("Cross Validation:OneVsRestClassifier: Decision Tree")

    for i in range(2):
        for j in range(5):
            title = "Depth: " + str(max_depth_vector[(i + 1) * (j + 1) - 1])
            axarr[i, j].plot(min_impurity_decrease_vector, scores[i * j][:])
            axarr[i, j].plot(min_impurity_decrease_vector, scores[i * j][:], 'x')

            axarr[i, j].set_title(title)
            axarr[i, j].grid()
            # axarr[i,j].set_ylabel("Score")

    plt.show()




