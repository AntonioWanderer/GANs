import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Sequential, Model

from keras.datasets import mnist

random_dim = 100
batch_size = 32
depth = 28 ** 2


def getModel():
    generator = Sequential()
    generator.add(Dense(256, activation='relu', input_dim=random_dim))
    generator.add(Dense(512, activation='relu'))
    generator.add(Dense(1024, activation='relu'))
    generator.add(Dense(depth, activation='tanh'))
    generator.compile(loss='binary_crossentropy')

    discriminator = Sequential()
    discriminator.add(Dense(1024, activation='relu', input_dim=depth))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(128, activation='relu'))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(64, activation='relu'))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy')

    gan_input = Input(shape=(random_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy')
    return generator, discriminator, gan


(x_train_loaded, y_train), (x_test, y_test) = mnist.load_data()
print(x_train_loaded.shape)
for digit in range(0,10):
    generator, discriminator, gan = getModel()
    ones = []
    for i in range(x_train_loaded.shape[0]):
        if y_train[i] == digit:
            ones.append(x_train_loaded[i, :, :])
    x_train = np.array(ones)
    print(x_train.shape)

    # noise = np.random.rand(batch_size, random_dim)
    image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)].reshape(batch_size, depth)
    res1 = np.ones(shape=(batch_size, 1))
    res0 = np.zeros(shape=(batch_size, 1))

    for i in range(1001):
        noise = np.random.normal(0, 1, size=[batch_size, random_dim])
        #fake
        generated_images = generator.predict(noise, verbose=False) #вот генерация
        discriminator.trainable = True #хорошо, проверим
        discriminator.train_on_batch(generated_images, res0) #вся ваша генерация - фейк!
        discriminator.train_on_batch(image_batch, res1)  # натуральные - не фейк
        discriminator.trainable = False #Проверил
        gan.train_on_batch(noise, res1) #генератор учится подделывать генерацию
        # discriminator.trainable = False
        # generator.train_on_batch(noise, image_batch)  # учится делать изображения
        # gan.train_on_batch(noise, res1) #пытается выдать за натуральное
        # discriminator.trainable = True
        # discriminator.train_on_batch(image_batch, res1)  # натуральные - натуральные
        # generator.trainable = False
        # gan.train_on_batch(noise, res0)  # сгенерированные - сгенерированные
        # generator.trainable = True
        # discriminator.trainable = False
        # gan.train_on_batch(noise, res1)  # пытается выдать за натуральное
        if i % 100 == 0:
            result = generator.predict(np.expand_dims(noise[0], axis=0))
            plt.imsave(fname=f"results/{digit}/result{i}.jpg", arr=result.reshape(28, 28))
            print(discriminator.predict(result))
