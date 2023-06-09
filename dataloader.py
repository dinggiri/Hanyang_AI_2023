import numpy as np
import torch
import pickle


def get_data_from_pickle(config):
	with open(config["root"] + config["train_feature"], 'rb') as fr:
	    features = pickle.load(fr)
	with open(config["root"] + config["train_target"], 'rb') as fr2:
	    targets = pickle.load(fr2)
	with open(config["root"] + config["test_feature"], 'rb') as fr:
	    features_test = pickle.load(fr)
	with open(config["root"] + config["test_target"], 'rb') as fr2:
	    targets_test = pickle.load(fr2)
	return features, targets, features_test, targets_test

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# Util Function
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    a = np.eye(num_classes)[y]
    return a

def make_train(features, targets, config):
	# append features
	X_train = []
	y_train = []
	for key in features:
	    X_train.append(features[key])
	    y_train.append(targets[key])

	# make it into array
	X_train = np.array(X_train)
	y_train = np.array(y_train)
	# make y_train into 1d vector
	tmp = np.squeeze(y_train)

	# make one-hot vector
	ys = []
	for t in tmp:
	    ys.append(config['labelencodedict'][t[0]])
	train_y = to_categorical(ys, num_classes=len(config["labelencodedict"].keys()))

	# make X_train into tensor
	X_train_numpy = X_train.astype(np.float32)
	X_train = torch.from_numpy(X_train_numpy)
	X_train = torch.Tensor(X_train)
	# make y_train into tensor
	y_train = torch.Tensor(ys)
	# train_y is one-hot vector
	train_y = torch.from_numpy(train_y)

	return X_train, y_train, train_y

def make_test(features_test, targets_test, config):
	# append features
	X_test = []
	y_test = []
	for key in features_test:
	    X_test.append(features_test[key])
	    y_test.append(targets_test[key])

	# make it into array
	X_test = np.array(X_test)
	y_test = np.array(y_test)

	# make y_test into 1d vector
	tmp = np.squeeze(y_test)

	# make one-hot vector
	ys_test = []
	for t in tmp:
	    ys_test.append(config['labelencodedict'][t[0]])
	test_y = to_categorical(ys_test, num_classes = len(config["labelencodedict"].keys()))


	# make X_test into tensor
	X_test_numpy = X_test.astype(np.float32)
	X_test =torch.from_numpy(X_test_numpy)
	X_test = torch.Tensor(X_test)
	# make y_test into tensor
	y_test = torch.Tensor(ys_test)
	# test_y is one-hot vector
	test_y = torch.Tensor(test_y)

	return X_test, y_test, test_y


       
def load_dataset(config):
	features, targets, features_test, targets_test = get_data_from_pickle(config)

	X_train, y_train, train_y = make_train(features, targets, config)
	X_test, y_test, test_y = make_test(features_test, targets_test, config)

	
	return X_train, y_train, train_y, X_test, y_test, test_y
