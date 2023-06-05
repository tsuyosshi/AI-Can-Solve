from flask import *
import cv2
import numpy as np
import keras
import pandas as pd
import json
import base64
from read_pazzle import *

# IMAGE_PATH = './image/slide/slide_4x4_001.jpg'

# filename = IMAGE_PATH
# with open(filename, "rb") as f:
#         img_base64 = base64.b64encode(f.read())

# print(img_base64)
# decoded = base64.b64decode(img_base64)
# jpg = np.asarray(bytearray(decoded), dtype=np.uint8)
# img = cv2.imdecode(jpg, 0)

img = np.load('./work/img.npy')

sz, df = predict_nums(img)

# print(df)

# cv2.imshow('window title', img)
# cv2.waitKey(0)

