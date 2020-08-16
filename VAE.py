import tensorflow.compat.v2 as tf
import numpy as np
import pandas as pd 
from PIL import Image
import math
import os

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("x_train:%s y_train:%s\nx_test:%s y_test:%s" %(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

class Visualizer:
    def __init__(self):
        self.exists = True

    def display_image(self,array):
        bestShape = self._getBestRectangle(array.shape)
        squared_array = array.reshape(bestShape)
        adjusted_squared_array = squared_array.transpose(0,2,1,3)
        merged_array = adjusted_squared_array.reshape(adjusted_squared_array.shape[0]*adjusted_squared_array.shape[1],adjusted_squared_array.shape[2]*adjusted_squared_array.shape[3])
        print("Image Layout: %s      Pixel Resolution: %s" %(bestShape, merged_array.shape))
        img = Image.fromarray(merged_array)
        img.show()

    def _getBestRectangle(self,shape):
        number = int(shape[0])
        if number%2 > 0:
            number -= 1
        root = math.sqrt(number)
        if int(root + 0.5) ** 2 == number:
            w = int(root)
            l = int(root)
        else:
            # What is the largest even value we can divide the number by where the remainder is also even and less than it
            w = 2
            l = 3
            answerl = None
            answerw = None
            while l > w:
                l = number/w
                if l == int(l):
                    answerl = int(l)
                    answerw = w
                w += 2
            l = answerl
            w = answerw
            # root = math.floor(root)
            # w = int(number/root)
            # l = int(number/w)
        return (w,l,shape[1],shape[2])
Viz = Visualizer()

def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')

x_train = preprocess_images(x_train)
x_test  = preprocess_images(x_test)

# Viz.display_image(x_test[:2000]*255)

train_dataset = (tf.data.Dataset.from_tensor_slices(x_train)).shuffle(60000).batch(32)

test_dataset = (tf.data.Dataset.from_tensor_slices(x_test)).shuffle(10000).batch(32)

encoder = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(28,28,1)),
        tf.keras.layers.Conv2D(
            filters=3, kernel_size=5, strides=(3,3), padding="same", activation="relu"
        )
    ]
)
batch = x_train[:100]
y = encoder(batch)
print(encoder.summary())
# print(batch.shape)
# print(y.shape)
# combined = np.concatenate((batch,y.numpy()),axis=0)*255
# print(combined.shape)
# Viz.display_image(combined)
Viz.display_image(y.numpy()*255)