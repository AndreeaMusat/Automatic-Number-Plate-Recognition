import operator
from scipy import misc
from imgproc import *
from loader import load_data
from skimage.measure import regionprops, label
from skimage.feature import hog
from skimage.transform import resize
from sklearn.neighbors import KNeighborsClassifier

IMAGE, BBOX = 0, 1
MIN_HEIGHT, MAX_HEIGHT = 0, 1
MIN_WIDTH, MAX_WIDTH = 2, 3
MIN_AREA = 4
LABEL, BBOX, PROB = 0, 1, 2


"""
Detect objects in an image by labelling the connected components and 
proposing the sub-images that might contain some object. 
This function is used for detecting the licence plate first and then 
for segmenting the characters from the number. For improved accuracy, 
some assumptions have been made: both plates and characters should 
have certain size limits, otherwise the objects will be discarded.

Parameters
----------
img : numpy.ndarray
	Image in which objects will be detected
size_limits : 5-tuple of Ints
	Minimum and maximum width and height and minimum area of the sub-image
debug : Boolean
	True if intermediate results should be saved
debug_file_name : string
	Name of the files in which intermediate results are to be saved

Returns
-------
result : [(numpy.ndarray, 4-tuple)]
	The result is a list of tuples. The first element of the tuple is the
	sub-image in which the object has been detected and the second element
	is the bounding box (relative to img)
"""
def detect_by_label(img, size_limits, debug=True, debug_file_name=None):
	edge_img = canny(img, blur=False, debug=debug)
	label_img = label(edge_img)
	regions = regionprops(label_img)

	result = []
	if debug:
		count = 0

	for region in regions:
		min_row, min_col, max_row, max_col = region.bbox
		bbox_height = max_row - min_row
		bbox_width = max_col - min_col

		if region.area < size_limits[MIN_AREA] or\
		   bbox_height < size_limits[MIN_HEIGHT] or\
		   bbox_height > size_limits[MAX_HEIGHT] or\
		   bbox_width < size_limits[MIN_WIDTH] or\
		   bbox_width > size_limits[MAX_WIDTH]:
			continue

		sub_img = img[min_row:max_row, min_col:max_col, :]
		result.append((sub_img, region.bbox))

		if debug:
			file_name = debug_file_name + "_" + str(count) + ".png"
			misc.imsave(file_name, sub_img)
			count = count + 1

	return result

"""
Compute hog features for an array of images and return them

Parameters
----------
data : numpy.ndarray
	Array of grayscale images

Returns
-------
features : List of numpy arrays
	Each numpy array represents the hog features of an image
"""
def get_features(data):
	features = []
	for image in data:
		feat, hog_image = hog(image, orientations=8, pixels_per_cell=(2, 2), \
									cells_per_block=(1, 1), block_norm='L2-Hys',\
									visualize=True, multichannel=False)
		features.append(feat)
	return features

"""
Load the data, create a K Nearest Neighbors Classifier and fit the data

Parameters
----------

Returns
-------
knn_classifier : KNeighborsClassifier
	Return the newly created classifier
"""
def get_symbol_classifier():
	train_data_dir = "./data/"
	tr_X, tr_y = load_data(train_data_dir)
	tr_feat = get_features(tr_X)
	knn_classifier = KNeighborsClassifier(n_neighbors=10, weights='uniform', n_jobs=4)
	knn_classifier.fit(tr_feat, tr_y)
	return knn_classifier

"""
Detect possible licence plate symbols

First, possible licence plates are detected using detect_by_label. Then, each
possible plate is segmented further into characters. A knn classifier is used 
for classifying the possible characters and a list of predictions is returned, 
one tuple per possible plate. Each tuple has 3 elements, each of them being a 
list: the predicted labels, the bounding boxes and the probabilities for 
each of them

Parameters
----------
input_img : numpy.ndarray
	Image in which the licence plate number should be detected

Returns
-------
predictions : List (of Lists of 3-tuples)
	See above
"""
def detect_number(input_img):

	plates_size_limits = (15, 50, 65, 180, 100)
	symbols_size_limits = (13, 30, 3, 40, 5)
	possible_plates = detect_by_label(input_img, plates_size_limits, \
							debug=True, debug_file_name="debug/plate")
	
	cnt_plate = 0

	knn_classifier = get_symbol_classifier()

	# this is an array of arrays (each prediction is an array)
	predictions = []

	for plate in possible_plates:
		possible_symbols = detect_by_label(plate[IMAGE], symbols_size_limits, \
								debug=True, debug_file_name="debug/symbol_" + str(cnt_plate))
		possible_symbols = sorted(possible_symbols, key = lambda x: x[BBOX][1])
		
		# any licence plate has at least 5 symbols
		if len(possible_symbols) < 5:
			continue

		# convert the images to grayscale and resize them
		data = [resize(grayscale(t[IMAGE]), (20, 20)) for t in possible_symbols]
		test_feat = get_features(data)

		predicted_labels = list(knn_classifier.predict(test_feat))
		predicted_bboxes = [t[BBOX] for t in possible_symbols]
		plate_coord_tuple = (plate[BBOX][0], plate[BBOX][1]) * 2
		predicted_bboxes = [tuple(map(operator.add, b, plate_coord_tuple)) for b in predicted_bboxes]
		predicted_probs = [max(l) for l in knn_classifier.predict_proba(test_feat)]

		predictions.append((predicted_labels, predicted_bboxes, predicted_probs))

		cnt_plate += 1

	return predictions

"""
Find the prediction with the best average probability.

Parameters
----------
predictions : List
	List of predictions computed using detect_number

Returns
-------
best_prediction : 3-tuple
	
"""
def find_best_prediction(predictions):
	best_prediction = None
	best_avg_prob = 0.0

	for prediction in predictions:
		avg_prob = 1.0 * sum(prediction[PROB]) / len(prediction[PROB])
		if avg_prob > best_avg_prob:
			best_prediction = prediction
			best_avg_prob = avg_prob

	labels = np.array(best_prediction[LABEL])
	bboxes = np.array(best_prediction[BBOX])
	probs = np.array(best_prediction[PROB])

	print(probs)

	labels = labels[np.where(probs >= 0.5)]
	bboxes = bboxes[np.where(probs >= 0.5)]
	probs = probs[np.where(probs >= 0.5)]
	best_prediction = (labels, bboxes, probs)
	
	return best_prediction