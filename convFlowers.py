import numpy as np
import cv2, os, time
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (Dense,
                                     Input,
                                     Dropout,
                                     Reshape,
                                     Conv2D,
                                     AvgPool2D,
                                     MaxPooling2D,
                                     Flatten)
from tensorflow.keras.models import Sequential, Model

path = "Content/flowers/"
x_train = []
for name in os.listdir(path):
    try:
        f = cv2.resize(cv2.imread(path + name), dsize=(128,128))
        #print(f.shape)
        x_train.append(f)
    except AttributeError:
        print("bad ",name)
    except cv2.error:
        print("empty", name)
x_train = np.array(x_train)
print(x_train.shape)

random_dim = 100
batch_size = 100
ims, height, width, dep = x_train.shape
depth = height * width * dep


def getModel():
    generator = Sequential()
    generator.add(Dense(units=300, activation='relu', input_dim=random_dim))
    generator.add(Dense(units=500, activation='relu', input_dim=random_dim))
    generator.add(Dense(units=1000, activation='relu', input_dim=random_dim))
    generator.add(Dense(units=1500, activation='relu', input_dim=random_dim))
    generator.add(Dense(units=2000, activation='relu', input_dim=random_dim))
    generator.add(Dense(depth, activation='tanh'))
    generator.add(Reshape(target_shape=(height,width,dep)))
    generator.compile(loss='binary_crossentropy')

    discriminator = Sequential()
    discriminator.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(height,width,dep)))
    discriminator.add(Flatten())
    discriminator.add(Dense(64))
    discriminator.add(Dense(32))
    discriminator.add(Dense(16))
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy')

    gan_input = Input(shape=(random_dim,))
    x = generator(gan_input)
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
    #time.sleep(5)
    image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
    noise = np.random.normal(0, 1, size=[batch_size, random_dim])
    generated_images = generator.predict(noise, verbose=False)  # вот генерация
    discriminator.trainable = True  # хорошо, проверим
    discriminator.train_on_batch(generated_images, res0)  # вся ваша генерация - фейк!
    discriminator.train_on_batch(image_batch, res1)  # натуральные - не фейк
    discriminator.trainable = False  # Проверил
    gan.train_on_batch(noise, res1)  # генератор учится подделывать генерацию
    if i%10 == 0:
        result = generator.predict(np.expand_dims(noise[0], axis=0)).reshape(height,width,dep)
        print(result.shape)
        cv2.imwrite(filename=f"results/flowers/result{i}.jpg", img=result * 255)
    if i % 500 == 0:
        generator.save(f"models/generator{i}")
        discriminator.save(f"models/discriminator{i}")
