import numpy as np
import cv2, os, time
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (Dense,
                                     Input,
                                     Dropout,
                                     Reshape,
                                     Conv2D,
                                     Conv2DTranspose,
                                     AvgPool2D,
                                     MaxPooling2D,
                                     BatchNormalization,
                                     Flatten)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import mnist
def getMnist(digit):
    (x_train_loaded, y_train), (x_test, y_test) = mnist.load_data()
    #print(x_train_loaded.shape)
    ones = []
    for i in range(x_train_loaded.shape[0]):
        if y_train[i] == digit:
            ones.append(x_train_loaded[i, :, :])
    x = np.array(ones)
    return x

path = "Content/Cat/"
#x_train = getMnist(1)
x_train = []
for name in os.listdir(path):
    try:
        f = cv2.resize(cv2.imread(path + name), dsize=(128, 128))
        # print(f.shape)
        x_train.append(f)
    except AttributeError:
        print("bad ", name)
    except cv2.error:
        print("empty", name)
x_train = (np.array(x_train)/128)-1
print(x_train.shape)
print(np.average(x_train))

random_dim = 100
batch_size = 100
#dep = 1
ims, height, width, dep = x_train.shape
depth = height * width * dep



def getModel():
    generator = Sequential()
    generator.add(Dense(units=(height - 2) * (width - 2) * dep, activation='relu', input_dim=random_dim))
    generator.add(Reshape(target_shape=(height - 2, width - 2, dep)))
    generator.add(Conv2DTranspose(filters=dep, kernel_size=(3, 3), activation='tanh'))
    generator.add(BatchNormalization())
    generator.compile(loss='mse')

    discriminator = Sequential()
    discriminator.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(height, width, dep)))
    discriminator.add(BatchNormalization())
    discriminator.add(MaxPooling2D())
    discriminator.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(height, width, dep)))
    discriminator.add(BatchNormalization())
    discriminator.add(MaxPooling2D())
    discriminator.add(Flatten())
    discriminator.add(Dense(32, activation="relu"))
    discriminator.add(Dense(16, activation="relu"))
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy')

    gan_input = Input(shape=(random_dim,))
    x = generator(gan_input)
    print(x.shape)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy')
    print(generator.summary())
    print(discriminator.summary())
    print(gan.summary())
    return generator, discriminator, gan


generator, discriminator, gan = getModel()

res1 = np.ones(shape=(batch_size, 1))
res0 = np.zeros(shape=(batch_size, 1))

for i in range(1000001):
    # time.sleep(5)
    image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
    noise = np.random.normal(0, 1, size=[batch_size, random_dim])
    generated_images = generator.predict(noise, verbose=False)  # вот генерация
    discriminator.trainable = True  # хорошо, проверим
    discriminator.train_on_batch(generated_images, res0)  # вся ваша генерация - фейк!
    discriminator.train_on_batch(image_batch, res1)  # натуральные - не фейк
    discriminator.trainable = False  # Проверил
    gan.train_on_batch(noise, res1)  # генератор учится подделывать генерацию
    #generator.train_on_batch(noise, image_batch) #генератор учится делать картинки
    if i % 10 == 0:
        result = 128*(generator.predict(np.expand_dims(noise[0], axis=0)).reshape(height, width, dep)+1)
        print(result.shape)
        cv2.imwrite(filename=f"results/Cat/result{i}.jpg", img=result)
        discriminator.evaluate(image_batch,res1)
        discriminator.evaluate(generated_images, res1)
        # fig, ax = plt.subplots(nrows=2, ncols=1)
        # ax.plot()
    if i % 500 == 0:
        generator.save(f"models/generator{i}")
        discriminator.save(f"models/discriminator{i}")
