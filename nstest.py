import numpy as np
import neuralStyle
try:
    from cv2 import imshow, waitKey
except:
    print("no cv2, use PIL to emulate cv2 and save output to output_*.png file")
    from PIL import Image as PIL_Image
    count = 0

    def imshow(title, img):
        global count
        im = PIL_Image.fromarray(img[:,:,::-1])
        im.save("output_%d.png"%count, format='png')
        count += 1
    def waitKey(n=0):
        pass

import sys

if len(sys.argv) < 3:
    print("python3 nstest.py photo_content_filename art_style_filename1  [art_style_filename2 ...]")
    print("Theano needs to compile functions, so proceesing time of first style will be longer")
    sys.exit(1)
photo_content_filename = sys.argv[1]
for art_style_filename in sys.argv[2:]:
    for img in neuralStyle.p_transfer(photo_content_filename, art_style_filename):
        imshow(art_style_filename, img)
        waitKey(1)
waitKey()
