import os
import random
from scipy import misc
from imgproc import grayscale

"""
Load training (and test, if necessary) input data and labels

Every input image should have its label char as the first char 
in its name. This function reads every image from the given 
directory and returns a 2-tuple or 4-tuple if test data is 
needed containing the grayscale images (not flattened) and 
their labels.

Parameters
----------
path : string
	The absolute path of the directory in which input data is found
test_data : Boolean
	True if test data is needed, False otherwise

Returns
-------
If test_data is False, returns a 2-tuple (X_train, y_train).
Otherwise, return a 4-tuple (X_train, y_train, X_test, y_test), 
where 20% of the data is randomly chosen as test data. Every 
element from X_train and X_test is numpy.ndarray image

"""
def load_data(path, test_data=False):
	images = []
	labels = []

	files = os.listdir(path)
	random.shuffle(files)

	for file in files:
		img = misc.imread(path + "/" + file)
		img = grayscale(img)
		images.append(img)
		labels.append(file[0])
	
	num_training_data = len(images)
	if test_data:
		num_training_data = int(0.8 * num_training_data)

	training_images = images[0:num_training_data]
	training_labels = labels[0:num_training_data]
	res = (training_images, training_labels)

	if test_data:
		test_images = images[num_training_data:]
		test_labels = labels[num_training_data:]
		res += (test_images, test_labels)

	return res