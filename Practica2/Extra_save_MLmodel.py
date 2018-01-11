'''-------------------------------------------------
Esto es un código de ejemplo para aprender a
guardar el modelo obtenido usando el módulo PICKLE
----------------------------------------------------'''

import numpy as np
import pandas as pd


# <b>This cell has the functions used all together</b><br>
# (Some may not be used in this script)

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train_set = data.iloc[train_indices]
    test_set  = data.iloc[test_indices]
    return train_set.reset_index(drop=True), test_set.reset_index(drop=True)

def scale_to_unit(data):
    data = (data / 255.0)
    return data

def attributes_to_features(data, theta=0.5):
    Ni,Na = data.shape
    #Ni = number of instances (rows)
    #Na = number of attributes(cols)
    set_features = []
    for ki in range(Ni):
        # get features out of every instance
        instance = np.reshape(data.ix[ki], [28,28])
        feature_vector_i = get_feature_vector(instance, theta=theta)
        set_features.append(feature_vector_i)
    return np.asarray(set_features)

def get_feature_vector(instance, theta):
    # index:   0       1      2      3      4      5       6      7
    # attrb: width , max1X, max2X, max3X, height, max1Y, max2Y, max3Y
    # ---features in X axis
    feature_vector = []
    # ---features in Y axis
    sum_x = np.sum(instance, axis=0)
    ind = np.argwhere(sum_x > theta * np.max(sum_x))
    width = ind[-1] - ind[0]
    feature_vector = np.append(feature_vector, width)
    ind_3max_x = np.argsort(sum_x)[-3:]
    feature_vector = np.append(feature_vector, ind_3max_x)
    # ---features in Y axis
    sum_y = np.sum(instance, axis=1)
    ind = np.argwhere(sum_y > theta * np.max(sum_y))
    height = ind[-1] - ind[0]
    feature_vector = np.append(feature_vector, height)
    ind_3max_y = np.argsort(sum_y)[-3:]
    feature_vector = np.append(feature_vector, ind_3max_y)
    return feature_vector

def feature_selection(set_features, selX, selY):
    pX = set_features[:, selX]
    pY = set_features[:, selY]
    return pX,pY

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

def from_RawData_to_Dataset(RawData_0,RawData_1,theta):

    # --- Scale the Train set ---------------------------
    RawData_0 = scale_to_unit(RawData_0)
    RawData_1 = scale_to_unit(RawData_1)

    # --- Get features from Train Set -------------------
    DataFeat_0 = attributes_to_features(RawData_0, theta)
    DataFeat_1 = attributes_to_features(RawData_1, theta)

    # --- Compile  data ---------------------------------
    # - we are full dim now !!
    data = np.vstack((DataFeat_0, DataFeat_1))
    labels = np.array([0] * len(DataFeat_0) + [1] * len(DataFeat_1))
    dataTable = {'set':data , 'data_0':DataFeat_0 , 'data_1':DataFeat_1 , 'tags':labels }
    return data, labels, dataTable

# <h3> == Main == </h3>

# <b>0. some options </b>


np.random.seed(seed=1234)
fraction_Test = 0.2
featX_index = 0
featY_index = 4
theta = 0.5

# <b>1.Build Train & Test sets </b>
#  <ol>
#   <li>Read csv</li>
#   <li>Scale. It is important not only to have all features in the same scale, and if possible within [0,1] </li>
#   <li>Feature engineering</li>
# </ol>

# --- Get data -------------------------------------
FullSet_0 = pd.read_csv('./1000ceros.csv', header=-1)
FullSet_1 = pd.read_csv('./1000unos.csv',  header=-1)

# --- Separate Test set -----------------------------
TrainSet_0, TestSet_0 = split_train_test(FullSet_0, fraction_Test)
TrainSet_1, TestSet_1 = split_train_test(FullSet_1, fraction_Test)

# --- Scale the Train set ---------------------------
X_train,tag_train,trainTable = from_RawData_to_Dataset(TrainSet_0,TrainSet_1,theta)


# <b> Random Forests </b>

# --- LEARN Random Forests --------------
from sklearn.ensemble import RandomForestClassifier

n_estimators = 500
max_leaf_nodes = 16
max_depth = 4

modelP2 = RandomForestClassifier(n_estimators=n_estimators,
                                max_depth=max_depth)
                                #max_leaf_nodes=max_leaf_nodes)

modelP2.fit(X_train, tag_train)

# <b> Save the model with "Pickel" module </b>

# --- SAVE the ML model -------------------------------

import pickle

# save the model to disk
Autor1 = 'Alfredo' #<- Poner aqui los nombres de los
Autor2 = 'SKlearn' #<-  dos autores.

filename = '%s_%s.model' % (Autor1, Autor2)
pickle.dump(modelP2, open(filename, 'wb'))


