import numpy as np


class load_data:
    # TODO: Preguntar a Alfredo si separar por categorias antes de separar por train,test,validation

    def split_train_test(self, data, test_ratio):
        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        train_set = data[train_indices]
        test_set = data[test_indices]
        return train_set, test_set

    def retag(self,tag):
        tag_array = np.zeros(len(self.tag_train))
        for i in range(0, len(self.tag_train)):
            if self.tag_train[i]==tag:
                tag_array[i]=1
        return tag_array

################################################
    def selection_data_from_tag(self,tag):

        tag_train_list = list(self.tag_train.astype(int))
        list_int = list(map(int,tag_train_list))
        indexA = [i for i,x in enumerate(list_int) if x== tag]
        indexB = [i for i,x in enumerate(list_int) if x!= tag]
        return self.train[indexA], self.train[indexB]
################################################
    def scale_to_unit(self, data, max):
        return np.array([data[:, itera] / max[itera] for itera in range(max.__len__())])

    def __init__(self,path, fraction_Test, fraction_Validation):

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
        self.test = test.T
        self.val = val.T
