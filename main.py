import numpy as np
from load_data import load_data
from matplotlib import pyplot as plt

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
    data2train = load_data('dataset_Facebook_selected_categories.csv',0.2,0.2)
    print(data2train.train)

    #data training
    clf = choose_classifier(option=1)
    clf.fit(data2train.train, data2train.tag_train)


    plt.plot(data2train.train[1], 'yo', alpha=.1)
    plt.plot(data2train.train[0], 'bx', alpha=.1)

    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(0, 20)
    yy = a * xx - (clf.intercept_[0] / w[1])

    plt.plot(xx, yy, 'r')
    strTitle = "w_X = %2.2f, w_Y = %2.2f, w_0 = %2.2f " % (w[0], w[1], clf.intercept_[0])
    plt.title(strTitle)
    #plt.axis([0, 20, 0, 20])

    plt.show()


