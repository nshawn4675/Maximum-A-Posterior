import random
import csv
import numpy as np


def csv2list(csv_file):
    """
    Parameters
    ----------
    csv_file : string
        the path of csv file.

    Returns
    -------
    data : list
        items with each line.
    """
    data = []
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def list2dataset(list_data):
    """
    Parameters
    ----------
    list_data : list
        many list data in a list.

    Returns
    -------
    dataset : list
        many dict data with y and X in a list.
    """
    dataset = []
    for data in list_data:
        data_dict = {}
        data_dict['y'] = data[0]
        data_dict['X'] = list(map(float, data[1:]))
        dataset.append(data_dict)
    return dataset

def csv2datasets(csv_file):
    """
    Parameters
    ----------
    csv_file : string
        csv file name.*

    Returns
    -------
    train_data : list
        training data with y and X.
    vali_data : list
        validation data with y and X.
    """
    list_data = csv2list(csv_file)
    dict_data = list2dataset(list_data)
    random.shuffle(dict_data)
    
    slice_idx = int(len(dict_data)/2)
    train_data = dict_data[:slice_idx]
    vali_data = dict_data[slice_idx:]
    
    return train_data, vali_data

class MAPmodel():
    def __init__(self, train_data):
        print("building model...")
        self.train_data = train_data
        self.Y = self.create_Y_list(self.train_data)
        self.X_len = len(self.train_data[0]['X'])
        self.classified_data = self.create_classified_data(self.Y,
                                                           self.train_data)
        self.pre_P = self.create_pre_P(self.Y, self.train_data)
        self.post_P = {}
        self.mean_vectors = self.create_mean_vectors(self.Y, self.X_len, 
                                                     self.train_data,
                                                     self.classified_data)
        self.cov_matrixes = self.create_cov_matrixes(self.Y,
                                                    self.classified_data)
        print("model built.")
        print()
    
    def create_classified_data(self, Y, train_data):
        """
        Parameters
        ----------
        Y : list
            list of training data labels.
        train_data : list
            list of training data.

        Returns
        -------
        classified_data : dict
            dict of data classified with labels.
        """
        classified_data = {}
        for y in Y:
            classified_data[y] = []
            for data in train_data:
                if data['y'] == y:
                    classified_data[y].append(data)
                    
        return classified_data
        
    def create_Y_list(self, train_data):
        """
        Parameters
        ----------
        train_data : list
            list of training data.

        Returns
        -------
        Y : list
            set of labels in training data.
        """
        Y = []
        for data in train_data:
            Y.append(data['y'])
            
        Y = set(Y)
        return Y
    
    def create_pre_P(self, Y, train_data):
        """
        Parameters
        ----------
        Y : list
            list of training data labels.
        train_data : list
            list of training data.

        Returns
        -------
        pre_P : dict
            dict of prior probability.
        """
        pre_P = {}
        for y in Y:
            pre_P[y] = 0
            
        for data in train_data:
            pre_P[data['y']] += 1
            
        total = len(train_data)
        for y in Y:
            pre_P[y] /= total
            
        return pre_P
    
    def create_mean_vector(self, X_len, train_data):
        """
        Parameters
        ----------
        X_len : int
            len of features.
        train_data : list
            list of training data.

        Returns
        -------
        mean_vector : list
            mean of each feature.
        """
        vectors = []
        for data in train_data:
            vectors.append(data['X'])
        vectors = np.array(vectors)
        mean_vector = [np.mean(vectors, axis=0).tolist()]
        mean_vector = np.array(mean_vector).T
        
        return mean_vector
    
    def create_mean_vectors(self, Y, X_len, train_data, classified_data):
        """
        Parameters
        ----------
        Y : list
            list of training labels.
        X_len : int
            len of features.
        train_data : list
            list of training data.

        Returns
        -------
        mean_vector : list
            mean of each feature.
        """
        mean_vectors = {}
        for y in Y:
            mean_vectors[y] = self.create_mean_vector(X_len, 
                                                      classified_data[y])
            
        return mean_vectors
    
    def create_cov_matrix(self, train_data):
        """
        Parameters
        ----------
        train_data : list
            list of training data.

        Returns
        -------
        cov_matrix : numpy-ndarray
            the covariance matrix of training data.
        """
        vectors = []
        for data in train_data:
            vectors.append(data['X'])
        vectors = np.array(vectors)
        vectors = vectors.T
        cov_matrix = np.cov(vectors, bias=True)
        return cov_matrix
    
    def create_cov_matrixes(self, Y, classified_data):
        """
        Parameters
        ----------
        Y : list
            list of training data labels.
        classified_data : dict
            dict of classified training data.

        Returns
        -------
        cov_matrixes : dict
            dict of classified covariance matrixes.
        """
        cov_matrixes = {}
        for y in Y:
            cov_matrixes[y] = self.create_cov_matrix(classified_data[y])
        
        return cov_matrixes
    
    def predict(self, target, X):
        """
        Parameters
        ----------
        target : string
            ground true of prediction.
        X : list
            features for prediction.

        Returns
        -------
        bool
            hit or miss.
        """
        X = np.array([X])
        X = X.T
        for y in self.Y:
            X_M = np.subtract(X, self.mean_vectors[y])
            X_M_T = X_M.T
            inv_cov_mat = np.linalg.inv(self.cov_matrixes[y])
            det_cov_mat = np.linalg.det(self.cov_matrixes[y])
            self.post_P[y] = -np.log(self.pre_P[y])\
                             + (1/2) * np.dot(np.dot(X_M_T, inv_cov_mat),X_M)\
                             + (1/2) * np.log(det_cov_mat)
            self.post_P[y] = self.post_P[y][0][0]
        min_label = min(self.post_P.keys(), key=(lambda i: self.post_P[i]))
        
        if min_label == target:
            return True
        else:
            print('min_label = {}'.format(min_label))
            print('target = {}'.format(target))
            print()
            return False
    
    def validation(self, vali_data):
        """
        Parameters
        ----------
        vali_data : list
            list of validation data.

        Returns
        -------
        acc : float
            accuracy of validation.
        """
        correct_cnt = 0
        print("wrong cases :")
        for data in vali_data:
            pred_res = self.predict(data['y'], data['X'])
            if pred_res:
                correct_cnt += 1
        
        print("correct count : {}".format(correct_cnt))
        print("total validation data : {}".format(len(vali_data)))
        print()
        acc = float(correct_cnt) / float(len(vali_data))
        return acc

    def details(self):
        """
        show model parameter information.
        
        Returns
        -------
        None.

        """
        print("labels : \n{}\n".format(self.Y))
        print("X length : {}\n".format(self.X_len))
        print("priori : \n{}\n".format(self.pre_P))
        print("mean vectors : \n{}\n".format(self.mean_vectors))
        print("covariance matrixes : \n{}\n".format(self.cov_matrixes))
        
if __name__ == '__main__':
    # data proprocessing
    csv_file = 'wine.data'
    train_data, vali_data = csv2datasets(csv_file)
    
    # build model using training data
    model = MAPmodel(train_data)
    # model.details()
    
    # validate model
    acc = model.validation(vali_data)
    print('Accuracy = {:.2f}%'.format(acc*100))