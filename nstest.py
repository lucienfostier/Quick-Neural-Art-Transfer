import numpy as np
import neuralStyle
import cv2
import sys

#art_style_filename = sys.argv[1]
photo_content_filename = sys.argv[1]
for art_style_filename in ["starry_night.jpg", "the_scream.jpg", "shipwreck.jpg", "picasso_selfport1907.jpg"]:
    art_style_filename = "styles/"+ art_style_filename
    for img in neuralStyle.p_transfer(photo_content_filename, art_style_filename):
        cv2.imshow(art_style_filename, img)
        cv2.waitKey(1)
cv2.waitKey()
