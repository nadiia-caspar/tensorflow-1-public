# My cource notes. Week 4

## Learn more...

[Gradient Descent in Practice II Learning Rate by Andrew Ng](https://www.youtube.com/watch?v=zLRB4oupj6g)

Now that you’ve learned how to download and process the horses and humans dataset, you’re ready to train. When you
defined the model, you saw that you were using a new loss function called [Binary Crossentropy](https://gombru.github.io/2018/05/23/cross_entropy_loss/),
and a new optimizer called [RMSProp](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop).
If you want to learn more about the type of binary classification we are doing here, check out [this great video](https://www.youtube.com/watch?v=eqEc66RFY0I&t=6s) from Andrew!

## ImageDataGenerator

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        './horse-or-human/',     # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=128,          # 
        class_mode='binary')      # Since we use binary_crossentropy loss, we need binary labels
```

## Loss function and optimizer

* **Binary Cross-Entropy (Log Loss): Used for binary classification problems.** - Loss function used for Classification Tasks

```python
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])
```
