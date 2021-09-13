# Description: This program classifies images
# Credit to: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
# And youtuber: Computer Science: https://www.youtube.com/watch?v=iGWbqhdjf2s 

# Imports
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Load the data
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Look at the data types of the variables
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))

# Get the shape of the arrays 
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

# Take a look at the first image as an array
index = 0
x_train[0]

# Take a look at the first image as an image
# img = plt.imshow(x_train[index])

# Get the image label
print('The image label is:', y_train[index])

# Get the image classification
classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# Print the image class
print('The image class is:', classification[y_train[index][0]])

# Convert the labels into a set of 10 numbers to input into the neural network
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Print the new labels
print(y_train_one_hot) # all col. will be 0 except the col which corresponds to the right label

# Print the new label of the image/picture above
print('The one hot label is:', y_train_one_hot[index])

# Normalize the pixels to be values between 0 and 1
x_train = x_train/255
x_test = x_test/255
# We devide with 255 due to the RGB numbers.

# Create the model's architecture
model = Sequential()

# Add the first layer (it will be a convelution layer)
model.add(Conv2D(32, (5,5), activation='relu', input_shape=(32,32,3)) )

# Add the second layer (it will be a pooling layer)
model.add( MaxPooling2D(pool_size=(2,2)))

# Add the third layer (it will be the second convelution layer)
model.add(Conv2D(32, (5,5), activation='relu') )

# Add the fourth layer (it will be the second pooling layer)
model.add( MaxPooling2D(pool_size=(2,2)))

# Add a fifth layer (it will be a flattening_layer)
model.add(Flatten())

# Add a sixth layer with 1000 neurons 
model.add(Dense(1000, activation='relu'))

# Add a drop out layer (50% dropout)
model.add(Dropout(0.5))

# Add a layer with 500 neurons 
model.add(Dense(500, activation='relu'))

# Add a drop out layer (50% dropout)
model.add(Dropout(0.5))

# Add a layer with 250 neurons 
model.add(Dense(250, activation='relu'))

# Add a layer with 10 neurons  (we have 10 different classifications)
model.add(Dense(10, activation='softmax'))


# Compile the model
model.compile(loss = 'categorical_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])


# Train the model:
hist = model.fit(x_train, y_train_one_hot,
                    batch_size=256,
                    epochs=2,
                    validation_split=0.2)


# Evaluate the model using the test data set
model.evaluate(x_test, y_test_one_hot)[1]
# What is the accuracy???:


# Visualize the model's accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy']) # val = validation
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Visualize the model's loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Test the model with an example:
new_image = plt.imread('download.jfif')
img = plt.imshow(new_image)
# Resize the image
from skimage.transform import resize
resized_img = resize(new_image, (32,32,3))
img = plt.imshow(resized_img)
plt.show()

# Classify image / prediction
predictions = model.predict(np.array([resized_img]))
# Show predictions
# ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# [[0.0316194  0.01439222 0.10035619 0.12316951 0.06245765 0.03335017, 0.54789454, 0.00444745 0.05357333 0.02873956]]
predictions
print(predictions)
# Sort the prediction
list_index = [0,1,2,3,4,5,6,7,8,9]
x = predictions

for i in range(10):
    for j in range(10):
        if x[0][list_index[i]] > x[0][list_index[j]]:
            temp = list_index[i]
            list_index[i] = list_index[j]
            list_index[j] = temp


# Show the sorted labels in order
print(list_index)
# Show the top 5 label
print(classification[list_index[0]], ':', round(predictions[0][list_index[0]] * 100, 2), '%')
print(classification[list_index[1]], ':', round(predictions[0][list_index[1]] * 100, 2), '%')
print(classification[list_index[2]], ':', round(predictions[0][list_index[2]] * 100, 2), '%')
print(classification[list_index[3]], ':', round(predictions[0][list_index[3]] * 100, 2), '%')
print(classification[list_index[4]], ':', round(predictions[0][list_index[4]] * 100, 2), '%')









