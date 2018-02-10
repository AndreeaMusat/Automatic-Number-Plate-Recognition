# Automatic-Number-Plate-Recognition

**Python packages used**__
1. numpy
2. scipy
3. skimage
4. sklearn

**Before usage:**__
unzip data.zip

**Usage:**__
python3 main.py input/in*.jpg

**Output:**__
output/detected_in*.png -> bounding boxes for the licence plate characters
output/result_in*.txt -> recognized chars and their probabilities

**Remarks**__
1. Only image processing and basic computer vision approaches have been used (+ knn classifier for digits/chars), so the script is not reliable in real life situations
2. If the licence plate is either too big or too small, it might not be recognized at all, as the script is discarding them if they do not fit in certain size limits.

