# My cource notes. Week 2

## Deep Neural Networks

**Deep Neural Network (DNN)** - is a type of artificial neural network (ANN) that consists of multiple layers of
interconnected nodes, called artificial neurons or units. The term "deep" in deep neural networks refers to the presence
of multiple hidden layers between the input and output layers. It is commonly used for tasks like image recognition,
natural language processing, and data generation.

**Sequential**: That defines a sequence of layers in the neural network.

**Flatten**: Remember earlier where our images were a 28x28 pixel matrix when you printed them out? Flatten just takes
that square and turns it into a 1-dimensional array.

**Dense**: Adds a layer of neurons. Each layer of neurons need an activation function to tell them what to do. There are
a lot of options, but just use these for now:

**ReLU**: only passes values greater than 0 to the next layer in the network. It effectively means:

```py
if x > 0: 
  return x

else: 
  return 0
```

**Softmax** takes a list of values and scales these so the sum of all elements will be equal to 1. When applied to model
outputs, you can think of the scaled values as the probability for that class. For example, in your classification model
which has 10 units in the output dense layer, having the highest value at index = 4 means that the model is most
confident that the input clothing image is a coat. If it is at index = 5, then it is a sandal, and so forth. See the
short code block below which demonstrates these concepts. You can also watch this lecture if you want to know more about
the Softmax function and how the values are computed.

**Adam (Adaptive Moment Estimation)** is an optimization algorithm, that combines the concepts of gradient descent and
momentum to effectively update the network's parameters during the training process. It adaptively adjust the learning
rate for each parameter based on their historical gradients. It keeps track of the first-order moment (the average of
the gradients) and the second-order moment (the average of the squared gradients) of each parameter. By using these
moments, Adam calculates adaptive learning rates for each parameter and performs updates accordingly.

**sparse_categorical_crossentropy** is a loss function commonly used for multi-class classification tasks when the target
labels are integers. It is suitable when the target variable is represented as a single integer per sample, rather than
one-hot encoded vectors. The loss function calculates the cross-entropy loss between the predicted probabilities and
the true class labels, helping to guide the model towards making accurate class predictions during training.

**input_shape=[1]** is a parameter used to define the shape of the input data in a TensorFlow model. It specifies the
dimensionality of the input data for a single sample or instance. For example, for the model in this lab, the input data
is a single value - 28x28 pixel matrix, which represents grayscale 28x28 pixel clothing images.

## Code example

```py
import tensorflow as tf
print(tf.__version__)

# Load the Fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist

# Load the training and test split of the Fashion MNIST dataset
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()


# Print the shape of the training images
import numpy as np
import matplotlib.pyplot as plt

## You can put between 0 to 59999 here
index = 530

## Set number of characters per row when printing
np.set_printoptions(linewidth=320)

## Print the label and image
print(f'LABEL: {training_labels[index]}')
print(f'\nIMAGE PIXEL ARRAY:\n {training_images[index]}')

## Visualize the image
plt.imshow(training_images[index])

# Normalize the pixel values of the train and test images
training_images  = training_images / 255.0
test_images = test_images / 255.0

# Build the classification model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Declare sample inputs and convert to a tensor
inputs = np.array([[1.0, 3.0, 4.0, 2.0]])
inputs = tf.convert_to_tensor(inputs)
print(f'input to softmax function: {inputs.numpy()}')

# Feed the inputs to a softmax activation function
outputs = tf.keras.activations.softmax(inputs)
print(f'output of softmax function: {outputs.numpy()}')

# Get the sum of all values after the softmax
sum = tf.reduce_sum(outputs)
print(f'sum of outputs: {sum}')

# Get the index with highest value
prediction = np.argmax(outputs)
print(f'class with highest probability: {prediction}')

# Compile the model
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
```

### E2Q1: Increase to 1024 Neurons -- What's the impact?

1. Training takes longer, but is more accurate
2. Training takes longer, but no impact on accuracy
3. Training takes the same time, but is more accurate

```py
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu), # Try experimenting with this layer
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
Epoch 1/5
1875/1875 [==============================] - 16s 8ms/step - loss: 0.4775
Epoch 2/5
1875/1875 [==============================] - 15s 8ms/step - loss: 0.3589
Epoch 3/5
1875/1875 [==============================] - 16s 8ms/step - loss: 0.3237
Epoch 4/5
1875/1875 [==============================] - 15s 8ms/step - loss: 0.2980
Epoch 5/5
1875/1875 [==============================] - 15s 8ms/step - loss: 0.2820
313/313 [==============================] - 1s 3ms/step - loss: 0.3352
313/313 [==============================] - 1s 3ms/step
[2.0968798e-08 9.2897437e-08 1.9451659e-07 3.2478893e-09 1.3300323e-07 1.6540046e-03 1.4429827e-06 2.6723951e-02 2.5486682e-07 9.7161990e-01]


model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu), # Try experimenting with this layer
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
Epoch 1/5
1875/1875 [==============================] - 27s 14ms/step - loss: 0.4702
Epoch 2/5
1875/1875 [==============================] - 25s 13ms/step - loss: 0.3549
Epoch 3/5
1875/1875 [==============================] - 28s 15ms/step - loss: 0.3209
Epoch 4/5
1875/1875 [==============================] - 25s 13ms/step - loss: 0.2955
Epoch 5/5
1875/1875 [==============================] - 25s 13ms/step - loss: 0.2770
313/313 [==============================] - 2s 5ms/step - loss: 0.3304
313/313 [==============================] - 1s 4ms/step
[3.7941820e-08 2.5322759e-08 1.7991520e-09 2.4520068e-09 5.2894644e-09 3.1491887e-04 2.5391514e-06 4.3464679e-02 1.9777929e-08 9.5621777e-01]
9
```

### E3Q1: What would happen if you remove the Flatten() layer. Why do you think that's the case?

It reinforces the rule of thumb that the first layer in your network should be the same shape as your data.

### Exercise 7:

Before you trained, you normalized the data, going from values that were 0-255 to values that were 0-1. What would be the impact of removing that? Here's the complete code to give it a try. Why do you think you get different results?

```py
# With
# training_images=training_images/255.0 # Experiment with removing this line
# test_images=test_images/255.0 # Experiment with removing this line
Epoch 1/5
1875/1875 [==============================] - 15s 8ms/step - loss: 0.4765
Epoch 2/5
1875/1875 [==============================] - 15s 8ms/step - loss: 0.3619
Epoch 3/5
1875/1875 [==============================] - 16s 8ms/step - loss: 0.3218
Epoch 4/5
1875/1875 [==============================] - 16s 9ms/step - loss: 0.2987
Epoch 5/5
1875/1875 [==============================] - 16s 9ms/step - loss: 0.2815
313/313 [==============================] - 1s 3ms/step - loss: 0.3637
313/313 [==============================] - 1s 3ms/step
[2.0121895e-06 1.5394855e-07 7.2675260e-07 6.7943589e-08 2.0638977e-07 2.5557359e-03 2.4760811e-06 1.8722136e-02 5.7755682e-07 9.7871590e-01]
9


# Without
# training_images=training_images/255.0 # Experiment with removing this line
# test_images=test_images/255.0 # Experiment with removing this line
Epoch 1/5
1875/1875 [==============================] - 18s 9ms/step - loss: 4.1037
Epoch 2/5
1875/1875 [==============================] - 15s 8ms/step - loss: 0.5313
Epoch 3/5
1875/1875 [==============================] - 15s 8ms/step - loss: 0.5014
Epoch 4/5
1875/1875 [==============================] - 15s 8ms/step - loss: 0.4842
Epoch 5/5
1875/1875 [==============================] - 15s 8ms/step - loss: 0.4785
313/313 [==============================] - 1s 3ms/step - loss: 0.5335
313/313 [==============================] - 1s 3ms/step
[5.8554453e-11 1.2273184e-08 9.7571433e-19 2.3023901e-08 1.6487815e-14 1.9018749e-02 4.4875229e-11 5.7924781e-02 6.2831087e-09 9.2305642e-01]
9
```

## Callback function

```py
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') >= 0.85): # Experiment with changing this value
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels) ,  (test_images, test_labels) = fmnist.load_data()

training_images=training_images/255.0
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])
```
