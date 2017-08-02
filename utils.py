from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np

def preprocess(observation, last_observation, width, height):
	processed_observation = np.maximum(observation, last_observation)
	processed_observation = np.uint8(resize(rgb2gray(processed_observation), (width, height)) * 255)
	return np.reshape(processed_observation, (width, height, 1))