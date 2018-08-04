from keras.models import load_model
import cv2
import numpy as np
import sys

# Get user supplied values
imagePath = sys.argv[1]

model = load_model('whole_model.h5')
# model = load_weights('model.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Read the image

img = cv2.imread(imagePath)
img = cv2.resize(img,(128,128))
img = np.reshape(img,[1,3,128,128])

classes = model.predict_classes(img)

print (classes)
