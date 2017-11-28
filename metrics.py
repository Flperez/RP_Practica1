import numpy as np

def confusion_matrix_multi(self, label_estimated, label_real, classes):
    matrix = np.zeros((classes, classes))
    for i in range(0, len(label_estimated)):
        matrix[label_real[i] - 1, label_estimated[i] - 1] += 1
    return matrix


def confusion_matrix_bi(self, label_estimated, label_real, classes):
    matrix = np.zeros((2, 2))
    for i in range(0, len(label_estimated)):
        matrix[label_real[i], label_estimated[i]] += 1
    return matrix

class Metrics:

    def __init__(self,matrix, category):

        # trueAcceptance
        trueAcceptance = matrix[category - 1, category - 1]

        # falseAcceptance
        for i in range(0,matrix.size[0]):
            falseAcceptance = falseAcceptance + matrix[i,i]
        falseAcceptance = falseAcceptance - matrix[category-1, category-1]

        # trueRejected
        sum_column = np.sum(matrix,axis=0)
        trueRejected = sum_column[0,category-1]- matrix[category-1, category-1]

        # falseRejected
        sum_row = np.sum(matrix, axis=1)
        falseRejected = sum_column[0,category-1]- matrix[category-1, category-1]

        FRR = falseRejected/(falseRejected - trueAcceptance)
        FAR = falseAcceptance/(falseAcceptance-trueRejected)
        Sensitivity = trueAcceptance/(falseRejected+trueAcceptance)
        Specify = trueRejected/(falseAcceptance+trueRejected)
        Precision = trueAcceptance/(trueAcceptance+falseAcceptance)
        Recall = trueAcceptance/(trueAcceptance+falseRejected)
        curveROC = [1-Specify, Sensitivity]
        curveRecall = [Recall,Precision]


        self.trueAcceptance = trueAcceptance
        self.falseAcceptance = falseAcceptance
        self.trueRejected = trueRejected
        self.falseRejected = falseRejected
        self.FRR = FRR
        self.FAR = FAR
        self.Sensitivity = Sensitivity
        self.Specify = Specify
        self.curveROC = curveROC
        self.curveRecall = curveRecall










