from sys import exit
from scipy import misc

def error(msg):
	print(msg)
	exit(-1)

"""
Draw a vertical line on an image

Parameters
----------
img : numpy.ndarray
	Image (RGB) to draw on
start_point : 2-tuple (Int)
	Coordinates of upper point
end_point : 2-tuple (Int)
	Coordinates of lower point
color : 3-tuple (Int)
	Color of the line 

Returns
-------
img : numpy.ndarray
	New image with a vertical line
"""
def draw_vertical_line(img, start_point, end_point, color):
	start_x, start_y = start_point
	end_x, end_y = end_point

	if start_y != end_y:
		error("Y coordinate mismatch")

	for i in range(start_x, end_x + 1):
		for ch in range(img.shape[2]):
			img[i][start_y][ch] = color[ch]

	return img

"""
Draw a horizontal line on an image

Parameters
----------
img : numpy.ndarray
	Image (RGB) to draw on
start_point : 2-tuple (Int)
	Coordinates of left point
end_point : 2-tuple (Int)
	Coordinates of right point
color : 3-tuple (Int)
	Color of the line

Returns
-------
img : numpy.ndarray
	New image with a horizontal line
"""
def draw_horizontal_line(img, start_point, end_point, color):
	start_x, start_y = start_point
	end_x, end_y = end_point

	if start_x != end_x:
		error("x coordinate mismatch")

	for i in range(start_y, end_y + 1):
		for ch in range(img.shape[2]):
			img[start_x][i][ch] = color[ch]

	return img

"""
Draw a rectangle on an image

Parameters
----------
img : numpy.ndarray
	Image (RGB) to draw on
upper_left_corner : 2-tuple (Int)
	Upper left corner of the rectangle
lower_right_corner : 2-tuple (Int)
	Lower right corner of the rectangle
color : 3-tuple (Int)
	Color of the line

Returns
-------
img : numpy.ndarray
	New image with a rectangle
"""
def draw_rectangle(img, upper_left_corner, lower_right_corner, color):
	min_x, min_y = upper_left_corner
	max_x, max_y = lower_right_corner
	img = draw_vertical_line(img, upper_left_corner, (max_x, min_y), color)
	img = draw_vertical_line(img, (min_x, max_y), lower_right_corner, color)
	img = draw_horizontal_line(img, upper_left_corner, (min_x, max_y), color)
	img = draw_horizontal_line(img, (max_x, min_y), lower_right_corner, color)

def draw_bounding_box(img, bbox, color):
	draw_rectangle(img, bbox[0:2], bbox[2:4], color)
