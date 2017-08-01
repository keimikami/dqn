from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers.core import Flatten, Dense
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv2D

def build_network(row, col, channels, num_class):
	model = Sequential()
	model.add(Conv2D(32, (8, 8), (4, 4), activation='relu', input_shape=(row, col, channels)))
	model.add(Conv2D(64, (4, 4), (2, 2), activation='relu'))
	model.add(Conv2D(64, (3, 3), (1, 1), activation='relu'))
	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dense(num_class))

	model.compile(optimizer='rmsprop',
		loss='categorical_crossentropy',
		metrics=['accuracy'])

	return model