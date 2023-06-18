import numpy as np
import torch
import pickle
from scipy import stats

###############################################################################################################
# Tried to make new dataloader with using, but it failed..... because the form is slightly different from this github
###############################################################################################################
## get from https://github.com/saif-mahmud/self-attention-HAR/blob/main/preprocess/pamap2/_sliding_window.py ##
###############################################################################################################

# def windowz(data, size, use_overlap=True):
#     start = 0
#     while start < len(data):
#         yield start, start + size
#         if use_overlap:
#             start += (size // 2)
#         else:
#             start += size

# def segment_pa2(x_train, y_train, train_y, window_size, n_sensor_val, n_classes):
# 	# train size :samples x channels x length, ex.21, 18, 10000
# 	# train_y size : samples ex. 21
# 	# y_train size : samples x channels ex. 21, 7
# 	# train_y : one-hot encoded
# 	x_train = x_train.view(x_train.shape[0], x_train.shape[0]*x_train.shape[1])
#		segments = np.zeros(((x_train.shape[-1] // (window_size // 2)) - 1, window_size, n_sensor_val))
#     labels = np.zeros(((x_train.shape[-1] // (window_size // 2)) - 1,))
#     labels_onehot = np.zeros(((x_train.shape[-1] // (window_size // 2)) - 1, n_classes))
#     i_segment = 0
#     i_label = 0
#     for idx, (start, end) in enumerate(windowz(x_train, window_size)):
#         if len(x_train[start:end]) == window_size:
#             # m = stats.mode(y_train[start:end])
#             segments[i_segment] = x_train[start:end]
#             labels[i_label] = idx
#             labels_onehot[i_label] = [1 if i==idx for i in range(n_classes) else 0]
#             i_label += 1
#             i_segment += 1
#     segments = np.transpose(segments, (0, 2, 1))
#     print("Done")
#     print(segments.shape)
#     print(labels.shape)
#     print(labels_onehot.shape)
#     return segments, labels, labels_onehot

# def segment_pa2_test(x_test, y_test, window_size, n_sensor_val, n_classes):
#     segments = np.zeros(((len(x_test) // (window_size)) + 1, window_size, n_sensor_val))
#     # labels = np.zeros(((len(y_test) // (wind
#     i_label = 0
#     # for (stow_size)) + 1))
#     for i in range(10):

#     # i_segment = 0art, end) in windowz(x_test, window_size, use_overlap=False):
#         if end >= x_test.shape[0]:
#             pad_len = window_size - len(x_test[start:end])
#             segments[i_segment] = x_test[start - pad_len:end]
#             m = stats.mode(y_test[start - pad_len:end])
#             labels[i_label] = m[0]
#         else:
#             m = stats.mode(y_test[start:end])
#             segments[i_segment] = x_test[start:end]
#             labels[i_label] = m[0]
#             i_label += 1
#             i_segment += 1

#     return segments, labels

# def segment_window_all(x_train, y_train, window_size, n_sensor_val, n_classes):
#     window_segments = np.zeros((len(x_train), window_size, n_sensor_val))
#     labels = np.zeros((len(y_train),))

#     total_len = len(x_train)

#     for i in range(total_len):
#         end = i + window_size

#         if end > total_len:
#             pad_len = end - total_len
#             window_segments[i] = x_train[i - pad_len:end]
#             labels[i] = y_train[total_len - 1]
#         else:
#             window_segments[i] = x_train[i:end]
#             labels[i] = y_train[end - 1]

#     return window_segments, labels

###############################################################################################################
##############################################  Implemented  ##################################################
###############################################################################################################
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

def get_data_from_pickle_with_validation(config):
	with open(config["root"] + config["train_feature"], 'rb') as fr:
		features = pickle.load(fr)
	with open(config["root"] + config["train_target"], 'rb') as fr2:
		targets = pickle.load(fr2)
	with open(config["root"] + config["val_feature"], 'rb') as fr:
		features_val = pickle.load(fr)
	with open(config["root"] + config["val_target"], 'rb') as fr2:
		targets_val = pickle.load(fr2)
	with open(config["root"] + config["test_feature"], 'rb') as fr:
		features_test = pickle.load(fr)
	with open(config["root"] + config["test_target"], 'rb') as fr2:
		targets_test = pickle.load(fr2)
	return features, targets, features_val, targets_val, features_test, targets_test

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

	X_train_revised = []
	for i in range(len(X_train)-1):
	    t = np.array([sub_array[:,1:20001] for sub_array in X_train[i]]).squeeze()      
	    X_train_revised.append(t)

	X_train = np.array(X_train_revised)
	X_train = torch.from_numpy(X_train.astype(float))
	tmp = y_train
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
	y_test = torch.Tensor(ys_test)

	return X_test, y_test, test_y

def make_val(features_val, targets_val, config):
	# append features
	X_val = []
	y_val = []
	for key in features_val:
	    X_val.append(features_val[key])
	    y_val.append(targets_val[key])

	X_val_revised = []

	for i in range(len(X_val)-1):
	    t = np.array([sub_array[:,1:20001] for sub_array in X_val[i]]).squeeze()      
	    X_val_revised.append(t)

	X_val = np.array(X_val_revised).astype(float)
	X_val = torch.from_numpy(X_val)
	tmp = y_val
	ys_val = []
	for t in tmp[:-1]:
	    ys_val.append(config['labelencodedict'][t[0]])

	val_y = to_categorical(ys_val, num_classes = len(config["labelencodedict"].keys()))
	val_y = torch.Tensor(val_y)
	y_val = torch.Tensor(ys_val)

	return X_val, y_val, val_y

def load_dataset(config):
	features, targets, features_test, targets_test = get_data_from_pickle(config)

	X_train, y_train, train_y = make_train(features, targets, config)
	X_test, y_test, test_y = make_test(features_test, targets_test, config)

	
	return X_train.to(config['device']), y_train.to(config['device']), train_y.to(config['device']), X_test.to(config['device']), y_test.to(config['device']), test_y.to(config['device'])


# def load_dataset_with_validation(config): # failed
# 	features, targets, features_val, targets_val, features_test, targets_test = get_data_from_pickle_with_validation(config)

# 	X_train, y_train, train_y = make_train(features, targets, config)
# 	X_val, y_val, val_y = make_train(features_val, targets_val, config)
# 	X_test, y_test, test_y = make_test(features_test, targets_test, config)

# 	n_sensor_val = config["first_in_channels"]
# 	X_train, y_train, train_y = segment_pa2(X_train, y_train, train_y, config['window_size'], n_sensor_val, len(config["labelencodedict"].keys()))
# 	X_val, y_val, val_y = segment_pa2(X_val, y_val, val_y, config['window_size'], n_sensor_val, len(config["labelencodedict"].keys()))
# 	X_test, y_test, test_y = segment_pa2(X_test, y_test, test_y, config['window_size'], len(config["labelencodedict"].keys()))

# 	return X_train.to(config['device']), y_train.to(config['device']), train_y.to(config['device']), X_val.to(config['device']), y_val.to(config['device']), val_y.to(config['device']), X_test.to(config['device']), y_test.to(config['device']), test_y.to(config['device'])