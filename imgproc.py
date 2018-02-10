import numpy as np
from scipy import misc

"""
Convert an image to grayscale by taking the median 
of the 3 RGB channels

Parameters
----------
img : numpy.ndarray
	Img to convert to grayscale

Returns
-------
out_img : numpy.ndarray
	Grayscale image


"""
def grayscale(img):
	if len(img.shape) == 2:
		return img

	out_img = np.mean(img, axis=-1)
	return out_img

"""
Simple image thresholding to convert image to B&W. 

The threshold is computed as the mean value of the pixels
in the image. All pixels having a value greater than the 
threshold will be white (255), while the others will be
black (0).

Parameters
----------
img : numpy.ndarray
	Input image (grayscale) to binarize

Returns
-------
res : numpy.ndarray
	B&W result image 
"""
def binarize(img):
	threshold = np.mean(img)
	res = np.copy(img)
	res[res >= threshold] = 255
	res[res < threshold] = 0
	return res

"""
Return a 2D Gaussian filter of given size and standard deviation

Parameters
----------
dim : Int
	Size of Gaussian filter (width and height)
std : Float
	Standard deviation 

Returns
-------
gaussian_filter : numpy.ndarray
	(dim * dim) matrix with values taken from a 2D Gaussian distribution

"""
def get_gaussian_filter(dim, std):
	if dim % 2 == 0:
		print("Filter size should be odd. Exiting")
		sys.exit(-1)

	k = (dim - 1) / 2
	gaussian_filter = np.array([[-((i - k)*(i - k) + (j - k) * (j - k)) \
							for i in range(dim)] for j in range(dim)])
	gaussian_filter = 1.0 / (2 * std * std * np.pi) * np.exp(gaussian_filter / (2 * std * std))
	return gaussian_filter

"""
Return the horizontal and vertical Sobel filters

Parameters
----------

Returns
-------

"""
def get_sobel_filters():
	horzontal_sobel_filter = np.array([[1], [ 2], [1]]) * np.array([1, 0, -1])
	vertical_sobel_filter = np.array([[1], [0], [-1]]) * np.array([1, 2, 1])
	return (horzontal_sobel_filter, vertical_sobel_filter)

"""
Apply filter to RGB or grayscale image

Parameters
----------
img : numpy.ndarray
	Input image
kernel : numpy.ndarray
	Filter to apply

Returns
-------
out_img : numpy.ndarray
	Result image 
"""
def apply_filter(img, kernel):
	if len(img.shape) > 2:
		height, width, channels = img.shape[0], img.shape[1], img.shape[2]
	else:
		height, width, channels = img.shape[0], img.shape[1], 1

	filter_size = kernel.shape[0]
	k = int((filter_size - 1) / 2)

	if channels > 1:
		padded_img = np.zeros((height + 2 * k, width + 2 * k, channels))
		for i in range(channels):
			padded_img[k:-k, k:-k, i] = img[:, :, i]
		
		out_img = [[[(padded_img[x:x+filter_size, y:y+filter_size, c] * kernel).sum() \
					for x in range(height)]\
					for y in range(width)]\
					for c in range(channels)]
		return np.array(out_img).T
	else:
		padded_img = np.zeros((height + 2 * k, width + 2 * k))
		padded_img[k:-k, k:-k] = img
		out_img = [[[(padded_img[x:x+filter_size, y:y+filter_size] * kernel).sum()\
					 for x in range(height)]\
					 for y in range(width)]]
		return np.array(out_img).T[:, :, 0]

"""
Perform an edge detection on a given RGB image and return 
the edges as a mask

Parameters
----------
input_img : numpy.ndarray
	RGB image stored in a numpy array
blue : Boolean
	True if a Gaussian filter should be applied to denoise the image

Returns
t1 : numpy.ndarray
	B&W mask containing the edges
-------
"""
def canny(input_img, blur=True, debug=False):
	edge_image = np.zeros(input_img.shape)

	# grayscale conversion
	grayscale_img = grayscale(input_img)

	# create Gaussian filter and apply it to the input image
	if blur:
		gaussian_filter = get_gaussian_filter(3, 0.5)
		blurred_img = apply_filter(grayscale_img, gaussian_filter)
		if debug:
			misc.imsave("debug/blurred_img.png", blurred_img)
	else:
		blurred_img = grayscale_img

	# now apply sobel filters to get horzontal (Gx) and vertical derivatives
	sobel_filters = get_sobel_filters()
	Gx = apply_filter(blurred_img, sobel_filters[0])
	Gy = apply_filter(blurred_img, sobel_filters[1])
	
	if debug:
		misc.imsave("debug/horizontal_derivative.png", Gx)
		misc.imsave("debug/vertical_derivative.png", Gy)

	# compute edge gradient and direction and save them
	G = np.sqrt(np.square(Gx) + np.square(Gy))
	theta = np.arctan2(Gy, Gx)
	
	if debug:
		misc.imsave("debug/edge_gradient.png", G)
		misc.imsave("debug/edge_direction.png", theta)

	# adjust the gradient direction
	theta = theta + 2 * np.pi
	theta = theta - (theta / (2 * np.pi)).astype(int) * 2 * np.pi
	theta[theta > np.pi] -= np.pi
	theta[(theta >= 0) & (theta < np.pi / 8)] = 0.0
	theta[(theta >= 7 * np.pi / 8) & (theta <= np.pi)] = 0.0
	theta[(theta >= np.pi / 8) & (theta < 3 * np.pi / 8)] = np.pi / 4
	theta[(theta >= 3 * np.pi / 8) & (theta < 5 * np.pi / 8)] = np.pi / 2
	theta[(theta >= 5 * np.pi / 8) & (theta < 7 * np.pi / 8)] = 3 * np.pi / 4

	if debug:
		misc.imsave("debug/adjusted_edge_direction.png", theta)

	directions = {}
	directions[0] = [(0, 1), (0, -1)]
	directions[3 * np.pi / 4] = [(-1, 1), (1, -1)]
	directions[np.pi / 2] = [(1, 0), (-1, 0)]
	directions[np.pi / 4] = [(-1, -1), (1, 1)]

	# non maxima suppression
	new_G = np.copy(G)
	for i in range(1, theta.shape[0] - 1):
		for j in range(1, theta.shape[1] - 1):
			neigh1 = (i + directions[theta[i, j]][0][0], j + directions[theta[i, j]][0][1])
			neigh2 = (i + directions[theta[i, j]][1][0], j + directions[theta[i, j]][1][1])
			if G[neigh1] >= G[i, j] or G[neigh2] >= G[i, j]:
				new_G[i, j] = 0
			elif G[i, j] > G[neigh1] and G[i, j] > G[neigh2]:
				new_G[neigh1] = 0
				new_G[neigh2] = 0
	
	if debug:
		misc.imsave("debug/thinned_edges.png", new_G)

	t1 = np.copy(new_G).astype(int)
	t1[0, :] = 0
	t1[-1, :] = 0
	t1[:, 0] = 0
	t1[:, -1] = 0
	t1[t1 >= 110] = 255
	t1[t1 < 110] = 0

	if debug:
		misc.imsave("debug/edges.png", t1)

	return t1
