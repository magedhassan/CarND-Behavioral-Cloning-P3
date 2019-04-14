import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Cropping2D

# Reading the csv file
data_folder = 'Behavioral-Cloning-Data/data'
lines = []
with open(data_folder+'/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Adjusting the path of the images and reading the images and their measurements
images = []
measurements = []
for line in lines[1:]:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'Behavioral-Cloning-Data/data/IMG/' + filename
        image = mpimg.imread(current_path)
        images.append(image)

    # correcting the measurement value for the left and right images
    correction = 0.2
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)

# Augmenting the simulation data by creating set of flipped images
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = float(measurement) * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

# Create numpy arrays to be used with Keras
X_train = np.asarray(augmented_images)
y_train = np.asarray(augmented_measurements)

# Test plotting random images
rand_index = random.randint(0, len(X_train)-1)
fig, axes = plt.subplots(1, 1)
fig.tight_layout()
axes.imshow(X_train[rand_index])
axes.set_title('steering angle: ' +str(y_train[rand_index]))

# Create a simple model
#model = Sequential()
#model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
#model.add(Flatten(input_shape=(160,320,3)))
#model.add(Dense(1))

# Create a LeNet architecture Model
#model = Sequential()
#model.add(Convolution2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D)
#model.add(Convolution2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D)
#model.add(Flatten())
#model.add(Dense(120))
#model.add(Dense(84))
#model.add(Dense(1))

# Create a NIVIDIA Model
model = Sequential()
# normalize the data
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
# crop the top and bottom parts of the images
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
# Define an Adam optimizer with a learning rate
optimizer = Adam(lr=0.001)
model.compile(loss='mse', optimizer=optimizer)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
model.save('model.h5')
exit()
