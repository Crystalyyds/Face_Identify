from tensorflow.keras import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense


def AlexNet(input_shape, num_classes):
    print("============AlexNet正常运行=================")
    model = Sequential()
    model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding="same", activation="relu"
                     , input_shape=input_shape, kernel_initializer="he_normal"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=None))
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=1, padding="same", activation="relu"
                     , kernel_initializer="he_normal"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=None))

    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding="same", activation="relu"
                     , kernel_initializer="he_normal"))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding="same", activation="relu"
                     , kernel_initializer="he_normal"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same", activation="relu"
                     , kernel_initializer="he_normal"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=None))

    model.add(Flatten())
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dense(units=4096, activation='relu'))

    model.add(Dense(units=num_classes, activation='softmax'))
    print("============AlexNet运行结束=================")

    return model
