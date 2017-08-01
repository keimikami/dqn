from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np

def preprocess(observation, width, height):
	processed_observation = np.uint8(resize(rgb2gray(observation), (width, height)) * 255)
	return np.reshape(processed_observation, (width, height, 1))