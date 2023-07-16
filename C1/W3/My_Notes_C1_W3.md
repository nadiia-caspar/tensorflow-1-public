# My cource notes. Week 3

## Learn more...

Learn more about convolutions
You’ve seen how to add a convolutional 2D layer to the top of your neural network in the previous video. If you want to
see more detail on how they work, check out the playlist at https://bit.ly/2UGa7uH. But do take note that it is not
required to complete this course.


## Convolutional Neural Networks

* **Convolution** - a process of applying a filter (or kernel) to an image. The filter is a square matrix of pixels that
is smaller than the original image. The filter moves across the image, one pixel at a time, performing mathematical
operations that produce a new pixel value for each location. The new image is called a **feature map**. It incorporates
convolutional layers that apply filters to input data, allowing the network to automatically learn and extract spatial
hierarchies of features.

**Pooling** - Pooling, also known as subsampling or down-sampling, is a technique used in convolutional neural networks
(CNNs) to reduce the spatial dimensions of feature maps. It helps to decrease the computational complexity of the network
and control overfitting.

**Padding** - Padding, in the context of convolutional neural networks (CNNs), refers to the process of adding extra
border pixels to the input data before applying convolutional operations. It is used to solve the following problems of
convolutional layers:

* srinking of the output data (feature map) compared to the input data
* loss of information on the boundaries of the input data

**model.summary()** - allow to inspect the layers of the model and see the joyrny of the image through the model.

**tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),** - it applies a set of learnable
filters to the input data, allowing the network to extract relevant features from the input. The parameters of the
convolutional layer are:

* filters: 32 is the number of filters - learnable convolutional filters or feature detectors applied to the input data.
The value here is purely arbitrary but it's good to use powers of 2 starting from 32.
* kernel_size: (3,3) is the size of the filter - the size of the filter is the size of the square matrix that will be
applied to.

**tf.keras.layers.MaxPooling2D(2, 2)** - max pooling is a downsampling technique that reduces the spatial dimensions of
the input while retaining the most salient features. The parameters of the max pooling layer are:

* pool_size: (2,2) is the size of the pooling window - the size of the pooling window is the size of the square matrix

**Overfitting** - Overfitting occurs when a machine learning model performs better on the training set than on the test.

**model.predict()**: This method is used to obtain predictions or outputs from a trained model given input data.
It takes input data as an argument and returns the predicted outputs based on the learned weights of the model.
The returned predictions can be used for further analysis or tasks such as classification, regression, or generating
new data.

**model.evaluate()**: This method is used to evaluate the performance of a trained model on a test dataset. It takes
test data and corresponding ground truth labels as arguments and computes various metrics specified during the model
compilation, such as accuracy, loss, or any other evaluation metric defined. It returns the computed metrics as
specified during model compilation.

In summary, **model.predict()** is used for obtaining predictions from a trained model on new or unseen data, while
**model.evaluate()** is used to evaluate the performance of the model by comparing its predictions with the ground truth
labels on a separate test dataset.

```python
import tensorflow as tf

# Load the Fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

# Normalize the pixel values
training_images = training_images / 255.0
test_images = test_images / 255.0

# Define the model
model = tf.keras.models.Sequential([
                                                         
  # Add convolutions and max pooling
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),

  # Add the same layers as before
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Print the model summary
model.summary()

# Use same settings
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
print(f'\nMODEL TRAINING:')
model.fit(training_images, training_labels, epochs=5)

# Evaluate on the test set
print(f'\nMODEL EVALUATION:')
test_loss = model.evaluate(test_images, test_labels)
```

## EXERCISES LAB 1

* Try editing the convolutions. Change the 32s to either 16 or 64. What impact will this have on accuracy and/or training time.

```python
# Convolutional Neural Network with 16 filters
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_2 (Conv2D)           (None, 26, 26, 16)        160                                                    
 max_pooling2d_2 (MaxPooling  (None, 13, 13, 16)       0
 conv2d_3 (Conv2D)           (None, 11, 11, 16)        2320
 max_pooling2d_3 (MaxPooling  (None, 5, 5, 16)         0         
 2D)
 flatten_2 (Flatten)         (None, 400)               0
 dense_4 (Dense)             (None, 128)               51328
 dense_5 (Dense)             (None, 10)                1290
=================================================================
Total params: 55,098
Trainable params: 55,098
Non-trainable params: 0
_________________________________________________________________

MODEL TRAINING:
Epoch 1/5
1875/1875 [==============================] - 8s 4ms/step - loss: 0.5157 - accuracy: 0.8130
Epoch 2/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.3475 - accuracy: 0.8753
Epoch 3/5
1875/1875 [==============================] - 7s 4ms/step - loss: 0.3034 - accuracy: 0.8888
Epoch 4/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2727 - accuracy: 0.8984
Epoch 5/5
1875/1875 [==============================] - 7s 4ms/step - loss: 0.2521 - accuracy: 0.9065

MODEL EVALUATION:
313/313 [==============================] - 1s 3ms/step - loss: 0.2790 - accuracy: 0.8994


# Convolutional Neural Network with 64 filters
Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_4 (Conv2D)           (None, 26, 26, 64)        640
 max_pooling2d_4 (MaxPooling  (None, 13, 13, 64)       0         
 2D)
 conv2d_5 (Conv2D)           (None, 11, 11, 64)        36928
 max_pooling2d_5 (MaxPooling  (None, 5, 5, 64)         0         
 2D)
 flatten_3 (Flatten)         (None, 1600)              0
 dense_6 (Dense)             (None, 128)               204928
 dense_7 (Dense)             (None, 10)                1290
=================================================================
Total params: 243,786
Trainable params: 243,786
Non-trainable params: 0
_________________________________________________________________

MODEL TRAINING:
Epoch 1/5
1875/1875 [==============================] - 9s 4ms/step - loss: 0.4393 - accuracy: 0.8398
Epoch 2/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2972 - accuracy: 0.8912
Epoch 3/5
1875/1875 [==============================] - 7s 4ms/step - loss: 0.2521 - accuracy: 0.9071
Epoch 4/5
1875/1875 [==============================] - 7s 3ms/step - loss: 0.2195 - accuracy: 0.9187
Epoch 5/5
1875/1875 [==============================] - 7s 4ms/step - loss: 0.1948 - accuracy: 0.9270

MODEL EVALUATION:
313/313 [==============================] - 1s 3ms/step - loss: 0.3018 - accuracy: 0.8943
```

**Conclusion**: 64 filters is better than 32 filters in accuracy, but does not show any significant difference on evaluation data.

* Remove the final Convolution. What impact will this have on accuracy or training time?

```python
MODEL TRAINING:
Epoch 1/5
1875/1875 [==============================] - 8s 3ms/step - loss: 0.3814 - accuracy: 0.8650
Epoch 2/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2579 - accuracy: 0.9064
Epoch 3/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2132 - accuracy: 0.9216
Epoch 4/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.1796 - accuracy: 0.9335
Epoch 5/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.1517 - accuracy: 0.9430

MODEL EVALUATION:
313/313 [==============================] - 1s 3ms/step - loss: 0.2749 - accuracy: 0.9113
```

**Conclusion**: time per epoch is reduced, but accuracy is increased.

* How about adding more Convolutions? What impact do you think this will have? Experiment with it.

**Conclusion**: accuracy is decreased

* Remove all Convolutions but the first. What impact do you think this will have? Experiment with it.
**Conclusion**: accuracy is increased

* In the previous lesson you implemented a callback to check on the loss function and to cancel training once it hit a
certain amount. See if you can implement that here.
  
## EXERCISES LAB 2
