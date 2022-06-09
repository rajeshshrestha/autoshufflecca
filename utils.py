import gzip
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import os
from copy import deepcopy
import torchvision.transforms as T



def load_mnist_data(data_file):
    """loads the data from the gzip pickled files, and converts to numpy arrays"""
    print('loading data ...')
    f = gzip.open(data_file, 'rb')
    train_set, valid_set, test_set = load_pickle(f)
    f.close()

    train_set_x, train_set_y = make_tensor(train_set)
    valid_set_x, valid_set_y = make_tensor(valid_set)
    test_set_x, test_set_y = make_tensor(test_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

def load_cifar_data(data_folder):
    '''load train data'''
    train_data_lst = []
    train_labels_lst = []
    for file_number in range(1,6):
        f = open(os.path.join(data_folder, f"data_batch_{file_number}"), 'rb')
        data_dict = load_pickle(f)
        data = torch.Tensor(data_dict['data'])
        labels = torch.Tensor(data_dict['labels'])
        train_data_lst.append(data)
        train_labels_lst.append(labels)
    
    all_train_data = torch.concat(train_data_lst, axis=0)
    all_train_labels = torch.concat(train_labels_lst, axis=0)

    '''split into train and validation data'''
    np.random.seed(10000)
    total_data_num = all_train_data.shape[0]
    valid_indices = np.random.randint(0, total_data_num, int(0.2*total_data_num))
    train_indices = list(set(range(total_data_num))-set(valid_indices))
    train_set_x, train_set_y = all_train_data[train_indices], all_train_labels[train_indices]
    valid_set_x, valid_set_y = all_train_data[valid_indices], all_train_labels[valid_indices]

    '''load test data'''
    f = open(os.path.join(data_folder, f"test_batch"), 'rb')
    data_dict = load_pickle(f)
    test_set_x = torch.Tensor(data_dict['data'])
    test_set_y = torch.Tensor(data_dict['labels'])

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]



def make_tensor(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    data_x = torch.tensor(data_x)
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y


def svm_classify(data, C):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    train_data, _, train_label = data[0]
    valid_data, _, valid_label = data[1]
    test_data, _, test_label = data[2]

    print('training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data, train_label.ravel())

    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)
    p = clf.predict(valid_data)
    valid_acc = accuracy_score(valid_label, p)

    return [test_acc, valid_acc]


def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddUniformNoise(object):
    def __init__(self, a=0., b=1.):
        self.a = a
        self.b = b
        
    def __call__(self, tensor):
        return tensor + torch.rand(tensor.size()) * (self.b-self.a) + self.a
    
    def __repr__(self):
        return self.__class__.__name__ + '(a={0}, b={1})'.format(self.a, self.b)


def normalize_cifar(data):
    transform = T.Compose([T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    for subdata_idx, subdata in enumerate(data):
        data_num,size = subdata[0].shape
        data[subdata_idx] = (transform(subdata[0].view(data_num, 3, int((size/3)**0.5),int((size/3)**0.5))).view(data_num,size), subdata[1])
    return data

def transform_image(tensor, channel_num):
    transform = T.Compose([
        # T.RandomHorizontalFlip(p=0.5),
        # T.RandomVerticalFlip(p=0.5),
        # T.RandomRotation((-90,90)),
        T.RandomAffine(degrees=0, scale=(0.95, 1.4)),
        AddGaussianNoise(10,50),
        # AddUniformNoise(a=-1,b=1),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

    data_num,size = tensor.shape
    return transform(tensor.view(data_num, channel_num, int((size/channel_num)**0.5),int((size/channel_num)**0.5))).view(data_num,size)


def get_normalized_agumented_data(data, channel_num=1):
    data2 = deepcopy(data)
    for subdata_idx, subdata in enumerate(data2):
        print(transform_image(subdata[0], channel_num))
        data2[subdata_idx] = (transform_image(subdata[0], channel_num), subdata[1])
    return data2
