import sys
import numpy as np
from scipy import misc
from detect import detect_number
from detect import find_best_prediction
from draw import draw_bounding_box

RED = (255, 0, 0)

def main(args):
	img = misc.imread(args[1])
	predictions = detect_number(img)
	
	if predictions == None or predictions == []:
		print("NONE")
		sys.exit(0)
	
	best_prediction = find_best_prediction(predictions)
	labels, bboxes, probs = best_prediction

	for i in range(len(labels)):
		draw_bounding_box(img, bboxes[i], RED)

	if "/" in args[1]:
		st = args[1].index("/")
	else:
		st = 0
	dt = args[1].index(".")

	result_name = args[1][st+1:dt]
	misc.imsave("output/detected_" + result_name + ".png", img)

	out_file = open("output/result_" + result_name + ".txt", "w")
	out_file.write(''.join(labels))
	out_file.write("\n" + str(list(np.round(probs, 2))))

	print("Detected licence plate number: ", ''.join(labels))

if __name__ == "__main__":
	main(sys.argv)