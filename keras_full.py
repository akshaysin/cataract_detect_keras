import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
import cv2 as cv

import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json

# Read image here in an list
path = "input_path"
listing = os.listdir(path)
count = size(listing)

# Validate size of an random image is 64*64
im_random = cv.imread("input_path//{0}".format(listing[23]))
im_random.shape

# Convert the image data into an imatrix. imatrix would just be a flat represetation of each image's pixel data in
# each row. This imatrix is whats gonna be used to create X_train, X_validation and X_test data.
imatrix = np.array([cv.imread("input_path//{0}".format(img)).flatten() for img in listing])
# imatrix = []
# for img in listing:
#     im_reading_now=cv.imread("input_path//{0}".format(img))
#     imatrix = np.vstack((imatrix,im_reading_now.flatten()))
print(imatrix.shape)
print(type(imatrix))
print(imatrix.ndim)

# Lets create y, i.e labels. This is whats gonna be used to create Y_train, Y_validation and Y_test data.
label = np.ones((count,), dtype=int)
label[0:1001] = 0
label[1001:2001] = 1
size(label)

# Lets randomnly suffle the data to avoid overfitting
data, label = shuffle(imatrix, label, random_state=2)
train_data = [data, label]

# Keras Parameters
batch_size = 32
nb_classes = 2
nb_epoch = 30
img_rows, img_col = 128, 128
img_channels = 3
nb_filters = 32
nb_pool = 2
nb_conv = 3

(X, y) = (train_data[0], train_data[1])
X.shape

# Splitting X and y in training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Splitting X_train and y_train in training and validation data
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=4)

# Validating the individual sizes
print("X_train : {0}".format(X_train.shape))
print("y_train :{0}".format(y_train.shape))

# print("X_val : {0}".format(X_val.shape))
# print("y_val : {0}".format(y_val.shape))

print("X_test : {0}".format(X_test.shape))
print("y_test : {0}".format(y_test.shape))

# Reshaping the data to pass to CNN
X_train = X_train.reshape(X_train.shape[0], 3, 128, 128)
# X_val = X_val.reshape(X_val.shape[0], 3, 128, 128)
X_test = X_test.reshape(X_test.shape[0], 3, 128, 128)

y_train = np_utils.to_categorical(y_train, nb_classes)
# y_val = np_utils.to_categorical(y_val, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

# Validating the individual sizes
print("X_train : {0}".format(X_train.shape))
print("y_train :{0}".format(y_train.shape))

# print("X_val : {0}".format(X_val.shape))
# print("y_val : {0}".format(y_val.shape))

print("X_test : {0}".format(X_test.shape))
print("y_test : {0}".format(y_test.shape))

# Regularize the data
X_train = X_train.astype('float32')
# X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
# X_val /= 255
X_test /= 255

# Define model now
model = Sequential()

model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                 padding="valid",
                 activation='relu',
                 input_shape=(img_channels, img_rows, img_col),
                 data_format='channels_first'))

model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Convolution2D(nb_filters, (nb_conv, nb_conv), activation='relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Convolution2D(nb_filters, (nb_conv, nb_conv), activation='relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.50))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
                    verbose=1, validation_data=(X_test, y_test))

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Test this trained model on our test data
# score = model.evaluate(X_test, y_test, verbose=1)
# print("Test Score :", score[0])
# print("Test accuracy: ", score[1])
# print(model.predict_classes(X_test[1:15]))
# print(y_test[1:15])

# Now lets save the model to disk
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
model.save("whole_model.h5")
print("Saved model to disk")

# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
