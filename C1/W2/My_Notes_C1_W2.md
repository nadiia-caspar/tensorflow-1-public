# My cource notes. Week 2


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
