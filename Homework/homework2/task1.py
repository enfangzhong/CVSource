from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D


def generate_vgg16():
    """
    搭建VGG16网络结构
    :return: VGG16网络
    """
    input_shape = (224, 224, 3)
    model = Sequential([
        Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(1000, activation='softmax')
    ])

    return model

if __name__ == '__main__':
    model = generate_vgg16()
    model.summary()