from keras import Sequential
from keras.src.layers import Conv2D


def AlexNet(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding="same", activation="relu"
                     , input_shape=input_shape, kernel_initializer="he_normal"))
