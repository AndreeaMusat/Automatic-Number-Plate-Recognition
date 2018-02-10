# Automatic-Number-Plate-Recognition

**Python packages used** <br />
1. numpy
2. scipy
3. skimage
4. sklearn

**Before usage:** <br />
unzip data.zip

**Usage:** <br />
python3 main.py input/in*.jpg

**Output:** <br />
output/detected_in*.png -> bounding boxes for the licence plate characters <br />
output/result_in*.txt -> recognized chars and their probabilities

**Remarks** <br />
1. Only image processing and basic computer vision approaches have been used (+ knn classifier for digits/chars), so the script is not reliable in real life situations
2. If the licence plate is either too big or too small, it might not be recognized at all, as the script is discarding them if they do not fit in certain size limits.

