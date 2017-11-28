import numpy as np


class load_data:

    def split_train_test(self, data, test_ratio):
        '''
        Dividimos los datos de entrada para training,validation y test
        :param data: datos de entrada
        :param test_ratio: ratio de division de los datos
        :return: conjunto de datos para training (train_set) y para test (test_set)
        '''
        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        train_set = data[train_indices]
        test_set = data[test_indices]
        return train_set, test_set

    def retag(self,tag):
        '''
        Funcion que recibe la etiqueta que deseamos tocar como 1 y modifica el
        el tag_train para solo seleccionar la etiqueta deseada
        :param tag: etiqueta que deseamos utilizar como 1
        :return: devuelve un vector con 0 y 1
        '''
        train_array = np.zeros(len(self.tag_train))
        for i in range(0, len(self.tag_train)):
            if self.tag_train[i]==tag:
                train_array[i]=1

        val_array = np.zeros(len(self.tag_val))
        for i in range(0, len(self.tag_val)):
            if self.tag_val[i] == tag:
                val_array[i] = 1

        test_array = np.zeros(len(self.tag_test))
        for i in range(0, len(self.tag_test)):
            if self.tag_test[i] == tag:
                test_array[i] = 1
        return train_array,val_array,test_array

################################################
    def selection_data_from_tag(self,tag,selection):
        '''

        :param tag:
        :return:
        '''


        #Queremos obtener los datos de entrenamiento
        if selection == "train":
            tag_train_list = list(self.tag_train.astype(int))
            list_int = list(map(int,tag_train_list))
            indexA = [i for i,x in enumerate(list_int) if x== tag]
            indexB = [i for i,x in enumerate(list_int) if x!= tag]
            dataA = self.train[indexA]
            dataB = self.train[indexB]

        if selection == "validation":
            tag_val_list = list(self.tag_val.astype(int))
            list_int = list(map(int, tag_val_list))
            indexA = [i for i, x in enumerate(list_int) if x == tag]
            indexB = [i for i, x in enumerate(list_int) if x != tag]
            dataA = self.val[indexA]
            dataB = self.val[indexB]

        if selection == "test":
            tag_test_list = list(self.tag_test.astype(int))
            list_int = list(map(int, tag_test_list))
            indexA = [i for i, x in enumerate(list_int) if x == tag]
            indexB = [i for i, x in enumerate(list_int) if x != tag]
            dataA = self.val[indexA]
            dataB = self.val[indexB]



        return dataA,dataB
################################################
    def scale_to_unit(self, data, max):
        '''
        Reescalado de 0-1
        :param data: datos
        :param max: array de los max para cada features
        :return: devuelve un np.array reescalado
        '''
        return np.array([data[:, itera] / max[itera] for itera in range(max.__len__())])

################################################

    def __init__(self,path, fraction_Test, fraction_Validation,selected_features):

        # Cargar datos
        data_load = np.genfromtxt(fname=path, delimiter=';', skip_header=1,
                                  usecols=(1, 2, 3, 4, 5))

        # Inicialización de valores
        max_caract1 = np.amax(data_load[:, 1])
        max_caract2 = np.amax(data_load[:, 2])
        max_caract3 = np.amax(data_load[:, 3])
        max_caract4 = np.amax(data_load[:, 4])
        max_caract_vector = [max_caract1, max_caract2, max_caract3, max_caract4]

        # Separar entre train, test validación
        train, test = load_data.split_train_test(self=self,data=data_load, test_ratio=fraction_Test)
        train, val = load_data.split_train_test(self=self,data=train, test_ratio=fraction_Validation)

        # Guardar primera columna ya que tiene la etiqueta
        tag_train = train[:, 0]
        tag_test = test[:, 0]
        tag_val = val[:, 0]

        # Quedarnos solo con las caracteristicas
        train = train[:, 1:5]
        test = test[:, 1:5]
        val = val[:, 1:5]

        # Normalizar caracteristicas
        train = load_data.scale_to_unit(self=self,data=train, max=max_caract_vector)
        test = load_data.scale_to_unit(self=self,data=test, max=max_caract_vector)
        val = load_data.scale_to_unit(self=self,data=val, max=max_caract_vector)

        #Asignar campos objeto
        self.tag_train = tag_train
        self.tag_test = tag_test
        self.tag_val = tag_val
        self.train = train.T

        self.train = train[selected_features,:].T
        self.test = test.T
        self.test = test[selected_features,:].T
        self.val = val.T
        self.val = val[selected_features,:].T




def confusion_matrix(label_estimated,label_real,classes):
    matriz=np.zeros((classes,classes))
    for i in range(0,len(label_estimated)):
        matriz[label_real[i],label_estimated[i]]+=1
    return matriz



