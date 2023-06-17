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

# Util Function
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    a = np.eye(num_classes)[y]
    return a

def make_train(features, targets, config):
	X_train = []
	y_train = []
	for key in features:
	    X_train.append(features[key])
	    y_train.append(targets[key])

	# print(X_train[-1])
	# y_train = np.array(y_train)
	# print(X_train)
	X_train_revised = []
	# print(X_train.shape[0])

	for i in range(len(X_train)-1):
	    # print(i)
	    
	    t = np.array([sub_array[:,1:20001] for sub_array in X_train[i]]).squeeze()      
	    # print(t)
	    X_train_revised.append(t)

	# print(X_train_revised)
	X_train = np.array(X_train_revised)
	X_train = torch.from_numpy(X_train.astype(float))
	# print(X_train.shape)
	tmp = y_train
	# tmp = np.squeeze(y_train)#.unsqueeze()
	ys = []
	for t in tmp[:-1]:
	    ys.append(config['labelencodedict'][t[0]])

	train_y = to_categorical(ys, num_classes=len(config["labelencodedict"].keys()))
	y_train = torch.Tensor(ys)
	train_y = torch.from_numpy(train_y)


	return X_train, y_train, train_y

def make_test(features_test, targets_test, config):
	# append features
	X_test = []
	y_test = []
	for key in features_test:
	    X_test.append(features_test[key])
	    y_test.append(targets_test[key])

	X_test_revised = []

	for i in range(len(X_test)-1):
	    t = np.array([sub_array[:,1:20001] for sub_array in X_test[i]]).squeeze()      
	    X_test_revised.append(t)

	X_test = np.array(X_test_revised).astype(float)
	X_test = torch.from_numpy(X_test)
	tmp = y_test
	ys_test = []
	for t in tmp[:-1]:
	    ys_test.append(config['labelencodedict'][t[0]])

	test_y = to_categorical(ys_test, num_classes = len(config["labelencodedict"].keys()))
	test_y = torch.Tensor(test_y)
	# X_test_numpy = X_test.astype(np.float32)
	# X_test =torch.from_numpy(X_test_numpy)
	# X_test = torch.Tensor(X_test)
	y_test = torch.Tensor(ys_test)

	# y_test = np.array(y_test)

	# # make y_test into 1d vector
	# tmp = np.squeeze(y_test)

	# # make one-hot vector
	# ys_test = []
	# for t in tmp:
	#     ys_test.append(config['labelencodedict'][t[0]])
	# test_y = to_categorical(ys_test, num_classes = len(config["labelencodedict"].keys()))


	# # make X_test into tensor
	# X_test_numpy = X_test.astype(np.float32)
	# X_test =torch.from_numpy(X_test_numpy)
	# X_test = torch.Tensor(X_test)
	# # make y_test into tensor
	# y_test = torch.Tensor(ys_test)
	# # test_y is one-hot vector
	# test_y = torch.Tensor(test_y)

	return X_test, y_test, test_y


       
def load_dataset(config):
	features, targets, features_test, targets_test = get_data_from_pickle(config)

	X_train, y_train, train_y = make_train(features, targets, config)
	X_test, y_test, test_y = make_test(features_test, targets_test, config)

	
	return X_train.to(config['device']), y_train.to(config['device']), train_y.to(config['device']), X_test.to(config['device']), y_test.to(config['device']), test_y.to(config['device'])
