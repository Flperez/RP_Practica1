import numpy as np
from load_data import load_data
from matplotlib import pyplot as plt
from load_data import confusion_matrix

#Fijar semilla
np.random.seed(seed=42)


def choose_classifier(option=4):
    if option == 1:
        from sklearn.linear_model import SGDClassifier
        clf = SGDClassifier(shuffle=True, n_iter=1000)
    elif option == 2:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
    elif option == 3:
        from sklearn.svm import SVC
        clf = SVC(kernel='linear')
    elif option == 4:
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(max_depth=8)
    else:
        print('choose one classifying method !')
        clf = []
    return clf



if __name__=='__main__':
    data2train = load_data('dataset_Facebook_selected_categories.csv',
                           fraction_Test = 0.2,
                           fraction_Validation = 0.2,
                           selected_features = (0,2))
    #print(data2train.train)
    features = data2train.train
    train_array,val_array,test_array = data2train.retag(3) #Biclase

    #data training
    clf = choose_classifier(option=1)
    clf.fit(features, train_array)
    featureA, featureB = data2train.selection_data_from_tag(tag=3,selection="train")

    #Plot training set with clf

    Ngrid = 100
    cmap = 'jet'  # 'jet' 'hot' 'cool' 'spring' 'summer' 'winter'
    threshold = 0  # Set 0 < threshold < 1 to make a hard classification
    rangeX = np.linspace(0, 0.5, Ngrid)
    rangeY = np.linspace(0, 0.5, Ngrid)
    xx, yy = np.meshgrid(rangeX, rangeY)
    xx = xx.reshape([xx.size, 1])
    yy = yy.reshape([yy.size, 1])
    grid_data = np.hstack((xx, yy))
    yhat = clf.decision_function((grid_data))
    if threshold > 0:
        t = (max(yhat) - min(yhat)) * threshold + min(yhat)
        yhat = (yhat > t)
    yhat = yhat.reshape([Ngrid, Ngrid])
    plt.contour(yhat, origin="lower", extent=[0, 1, 0, 1], cmap=cmap, alpha=0.3)

    plt.plot(featureA[:,0], featureA[:,1], 'yo', alpha=.15)
    plt.plot(featureB[:,0], featureB[:,1], 'bx', alpha=.15)

    strTitle = "Non-linear classifier"
    plt.title(strTitle)
    plt.axis([0, 1, 0, 1])  # <--This axis are set for features number 0(featX_index) and 4(featY_index).
    #    If you pick another two it is likely to need another axis
    plt.show()

    # plt.plot(featureA[:,1],featureA[:,0], 'yo', alpha=.1)
    # plt.plot(featureB[:,1],featureB[:,0], 'bx', alpha=.1)
    #
    # w = clf.coef_[0]
    # a = -w[0] / w[1]
    # xx = np.linspace(0,1)
    # yy = a * xx - (clf.intercept_[0] / w[1])
    #
    # plt.plot(xx, yy, 'r')
    # strTitle = "w_X = %2.2f, w_Y = %2.2f, w_0 = %2.2f " % (w[0], w[1], clf.intercept_[0])
    # #strTitle ='a'
    # plt.title(strTitle)
    # #plt.axis([0, 20, 0, 20])
    #
    # plt.show()



    # Validation set
    featureA_val, featureB_val = data2train.selection_data_from_tag(tag=3,selection="train")

