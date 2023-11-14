import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, kernel_size=3, activation='relu')) 
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Train the model
checkpoint = ModelCheckpoint('model.h5', save_best_only=True) 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=3, 
                    validation_data=(x_test, y_test), callbacks=[checkpoint])

# Evaluate the model
model.load_weights('model.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plot training & validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Generate confusion matrix
predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = tf.math.confusion_matrix(y_true, y_pred)
print(cm)

# Make predictions on new data
# Allow user to upload an image
# Preprocess image
# Make prediction
# Print prediction result

import numpy as np
from keras.preprocessing import image

# Allow user to upload an image 
from tkinter import filedialog
from PIL import Image
filepath = filedialog.askopenfilename()
img = Image.open(filepath)

# Preprocess image
img = img.resize((32,32))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x/255.0

# Make prediction
prediction = model.predict(x)
index = np.argmax(prediction)

# Print prediction result
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print("Prediction: ", classes[index])

