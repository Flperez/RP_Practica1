import numpy as np
from load_data import load_data
#Fijar semilla
np.random.seed(seed=42)

if __name__=='__main__':
    data2train = load_data('dataset_Facebook_selected_categories.csv',0.2,0.2)
    #print(data2train.tag_test)