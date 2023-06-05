import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageFilter
from tqdm import tqdm

def load_img(filename):
    image = Image.open(filename)
    image = image.convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(28*28)
    image = image / 255.0
    return image