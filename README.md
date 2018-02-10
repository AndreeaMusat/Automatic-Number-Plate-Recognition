# Automatic-Number-Plate-Recognition

**Python packages used**
1. numpy
2. scipy
3. skimage
4. sklearn

**Before usage**
unzip data.zip

**Usage**
python3 main.py input/in*.jpg

**Output**
output/detected_in*.png -> bounding boxes for the licence plate characters
output/result_in*.txt -> recognized chars and their probabilities

**Remarks**
If the licence plate is either too big or too small, it might not be recognized at all, as the script is discarding them if they do not fit in certain size limits.
