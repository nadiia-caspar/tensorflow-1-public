# Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning (Course 1 of the TensorFlow Developer Specialization)

## My cource notes. Week 1

### Where to find the notebooks for this course

All notebooks in this course can be run in either Google Colab or Coursera Labs. You don’t need a local environment set
up to follow the coding exercises. You can simply click the Open in Colab badge at the top of the ungraded labs while
for the assignments, you will be taken automatically to Coursera Labs.

However, if you want to run them on your local machine, the ungraded labs and assignments for each week can be found in this 
Github repository
 under the C1 folder. If you already have git installed on your computer, you can clone it with this command:

```shell
git clone https://github.com/https-deeplearning-ai/tensorflow-1-public
```

If not, please follow the guides [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to install git
on your operating system. Once you’ve cloned the repo, you can do a git pull once in a while to make sure that you get
the latest updates to the notebooks.

You will need these packages if you will run the notebooks locally:

```python
tensorflow==2.7.0
scikit-learn==1.0.1
pandas==1.1.5
matplotlib==3.2.2
seaborn==0.11.2
```

### Learn more...

* AI For Everyone is a non-technical course that will help you understand many of the AI technologies we will discuss
later in this course, and help you spot opportunities in applying this technology to solve your problems.
<https://www.deeplearning.ai/ai-for-everyone/>

TensorFlow is available at [TensorFlow.org](https://www.tensorflow.org/), and video updates from the TensorFlow team
are at [youtube.com/tensorflow](https://www.youtube.com/tensorflow).

Play with a neural network right in the browser at [http://playground.tensorflow.org](https://playground.tensorflow.org/).

### Cource notes

* **Convergence** - A machine learning model reaches convergence when it achieves a state during training in which loss
settles to within an error range around the final value. In other words, a model converges when additional training will
not improve the model.
  * A process of getting very close to the correct answer.

* **Loss function** - A loss function, also known as a cost function, takes into account the probabilities or
uncertainty of a prediction based on how much the prediction varies from the true value.  Here are some common loss
functions for different types of machine learning tasks:

  * **Regression Tasks**: Regression tasks in machine learning involve predicting a continuous numerical value or a real-valued output based on input features. In these tasks, the goal is to learn a function that maps the input data to a continuous target variable. The output is not limited to a specific set of discrete classes but rather spans a continuous range of values. Common examples of regression tasks include: House Price Prediction, Stock Price Forecasting, Temperature Prediction, Demand Forecasting.

    * Mean Squared Error (MSE): The average of the squared differences between the predicted and actual target values.

      ```python
      model.compile(optimizer='sgd',
              loss='mean_squared_error')
      ```

    * Mean Absolute Error (MAE): The average of the absolute differences between the predicted and actual target values.

    * Huber Loss: A combination of MSE and MAE, providing a balance between the two.

  * **Classification Tasks**: Classification tasks in machine learning involve predicting discrete categories or classes for a given input data point. In these tasks, the model learns to map the input features to a specific class label based on the patterns present in the training data. The output is a discrete value representing the class membership of the input, and the goal is to achieve accurate predictions for new, unseen data. Common examples of classification tasks include: Email Spam Detection, Image Classification (cat, not-cat), Sentiment Analysis, Medical Diagnosis, Handwriting Recognition, Language Identification.

    * Binary Cross-Entropy (Log Loss): Used for binary classification problems.

      ```python
      model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])
      ```

    * Categorical Cross-Entropy (Softmax Cross-Entropy): Used for multi-class classification problems, where each input belongs to one of multiple classes.

    * Sparse Categorical Cross-Entropy: Similar to categorical cross-entropy, it is used for multi-class classification tasks when the target labels are integers instead of one-hot encoded vectors.

      ```python
      model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
      ```
  
  * Ranking Tasks:

    * Hinge Loss (SVM Loss): Used in support vector machines and other ranking-based algorithms.

    * Pairwise Ranking Loss: Used for learning-to-rank tasks, comparing pairs of examples.

  * Custom Loss Functions:

    * Depending on the specific requirements of a problem, researchers and practitioners sometimes design custom loss functions tailored to their particular applications.

* **Optimizer** - is an algorithm or a method used to update the parameters (weights and biases) of a neural network during the training process. The goal of training a deep learning model is to find the optimal set of parameters that minimizes the chosen loss function and improves the model's performance on the given task.
Different optimizers use various update rules and strategies to adjust the parameters. Some of the commonly used optimizers in deep learning include:

  * **Stochastic Gradient Descent (SGD):** The simplest and most fundamental optimizer that updates the parameters in the direction of the negative gradient of the loss function. It typically uses a small subset (mini-batch) of the training data at each iteration.

      ```python
      model.compile(optimizer='sgd',
              loss='mean_squared_error')
      ```

  * **Adam (Adaptive Moment Estimation):** An adaptive learning rate optimization algorithm that combines the benefits of RMSprop and Momentum. It adapts the learning rate for each parameter based on the past gradients.

    ```python
    model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy')
    ```

  * **RMSprop (Root Mean Square Propagation):** An adaptive learning rate optimizer that adjusts the learning rate for each parameter based on the magnitude of past gradients.

    ```python
    model.compile(loss='binary_crossentropy',
            optimizer=RMSprop(learning_rate=0.001),
            metrics=['accuracy'])
    ```

  * **Adagrad (Adaptive Gradient Algorithm):** An adaptive learning rate optimizer that scales the learning rate for each parameter based on the historical squared gradients.

  * **Adadelta:** An extension of Adagrad that addresses its learning rate's diminishing problem by using an exponentially decaying average of past squared gradients.

  * **RAdam (Rectified Adam):** A variant of Adam that improves convergence by rectifying the variance of the adaptive learning rates.

* **Labelling the Data** - the process of telling a computer what the data represents (i.e. his data is for walking, his data is for running).

* **Activation function**: is a non-linear transformation applied to the output of a neuron or a layer in a neural network.
Activation functions introduce non-linearity to the network, enabling it to learn complex relationships and patterns in
the data. In a neural network layer, each neuron takes the weighted sum of its inputs, adds a bias term, and then applies
the activation function to produce the output of that neuron. Some common activation functions available in Keras are:

  * **ReLU (Rectified Linear Unit):** Defined as f(x) = max(0, x), ReLU sets all negative values to zero and leaves positive values unchanged. It is widely used in many deep learning architectures due to its simplicity and effectiveness. [ReLU](https://keras.io/api/layers/activations/#relu-function) effectively means:

    ```python
    if x > 0: 
      return x

    else: 
      return 0
    ```

    In other words, it only passes values greater than 0 to the next layer in the network.

      ```python
      model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                      tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                      tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
      ```

  * **Sigmoid:** The sigmoid function is defined as f(x) = 1 / (1 + exp(-x)). It squashes the output between 0 and 1, making it suitable for binary classification problems.

    ```python
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 300x300 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        ...
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    ```

  * **Tanh (Hyperbolic Tangent):** The tanh function is defined as f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)). It maps the output between -1 and 1, which can be useful for certain types of problems.

  * **Softmax:** The softmax function is often used in the output layer of multi-class classification problems. It converts the raw output scores into probabilities, ensuring that the sum of probabilities for all classes is equal to 1. [Softmax](https://keras.io/api/layers/activations/#softmax-function) takes a list of values and scales these so the sum of all elements will be equal to 1. When applied to model outputs, you can think of the scaled values as the probability for that class. For example, in your classification model which has 10 units in the output dense layer, having the highest value at `index = 4` means that the model is most confident that the input clothing image is a coat. If it is at index = 5, then it is a sandal, and so forth. See the short code block below which demonstrates these concepts. You can also watch this [lecture](https://www.youtube.com/watch?v=LLux1SW--oM&ab_channel=DeepLearningAI) if you want to know more about the Softmax function and how the values are computed.

    ```python
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    ```

* What does the optimizier do:
  * Generates a new and improved guess

## My cource notes. Week 2

### Deep Neural Networks

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

### Learn more...

The structure of Fashion MNIST data
Here you saw how the data can be loaded into Python data structures that make it easy to train a neural network. You saw
how the image is represented as a 28x28 array of greyscales, and how its label is a number. Using a number is a first
step in avoiding bias -- instead of labelling it with words in a specific language and excluding people who don’t speak
that language! You can learn more about bias and techniques to avoid it [here](https://ai.google/responsibility/responsible-ai-practices/).

[Neural Networks and Deep Learning (Course 1 of the Deep Learning Specialization) - Neural Network Overview (C1W3L01)](https://youtu.be/fXOsFF95ifk)