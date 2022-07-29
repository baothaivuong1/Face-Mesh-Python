import cv2 as cv
import numpy as np
from PIL import ImageFont, ImageDraw, Image

image = cv.imread('img/img_001.png')

cv.imshow('test',image)
cv.waitKey(0)
cv.destroyAllWindows()
