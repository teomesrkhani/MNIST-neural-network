import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

# Implementing a neural network using Keras using the book: 
# Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron

# load the MNIST digits classification dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Visualize the first 10 instances (digits) from the dataset
plt.figure(figsize=(5,2))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.axis('off')  
plt.show()

# Scale the input feature down to 0-1 values, by dividing them by 255.0 
x_train = x_train / 255.0
x_test = x_test / 255.0

# Creating a Sequential model
model = keras.models.Sequential()

# First layer to the model 
model.add(keras.layers.Flatten(input_shape=[28, 28]))

# First hidden layer to the model, using 100 neurons
model.add(keras.layers.Dense(300, activation="relu"))

# Second hidden layer to the model, using 100 neurons
model.add(keras.layers.Dense(100, activation="relu"))

# Output layer to the model, with 10 output neurons, using softmax activation function
# because we have a multi-classification problem.
model.add(keras.layers.Dense(10, activation="softmax"))

# Display the model’s layers
model.summary()

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd")

# Train the model for 20 epochs
model.fit(x_train, y_train, epochs=20)



# Test the model on the first 10 instances of the test set
plt.close('all')
y_pred = model.predict(x_test[:10])
plt.figure(figsize=(5,2))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.title('Predicted label: ' + str(np.argmax(y_pred[i])))
    plt.imshow(x_test[i], cmap='gray')
    plt.axis('off')  
plt.show()
