import numpy as np
import cv2, os
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Sequential, Model

path = "Content/flowers/"
x_train_l = np.array([cv2.imread(path + name) for name in os.listdir(path)])
print(x_train_l.shape)

random_dim = 1000
batch_size = 1
ims, height, width, dep = x_train_l.shape
depth = height * width


def getModel():
    generator = Sequential()
    generator.add(Dense(256, activation='relu', input_dim=random_dim))
    generator.add(Dense(512, activation='relu'))
    generator.add(Dense(1024, activation='relu'))
    generator.add(Dense(depth, activation='tanh'))
    generator.compile(loss='binary_crossentropy')

    discriminator = Sequential()
    discriminator.add(Dense(256, activation='relu', input_dim=depth))
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


x_train1 = x_train_l[:, :, :, 0]
x_train2 = x_train_l[:, :, :, 1]
x_train3 = x_train_l[:, :, :, 2]
generator1, discriminator1, gan1 = getModel()
generator2, discriminator2, gan2 = getModel()
generator3, discriminator3, gan3 = getModel()

image_batch1 = x_train1[np.random.randint(0, x_train1.shape[0], size=batch_size)].reshape(batch_size, depth)
image_batch2 = x_train2[np.random.randint(0, x_train2.shape[0], size=batch_size)].reshape(batch_size, depth)
image_batch3 = x_train3[np.random.randint(0, x_train3.shape[0], size=batch_size)].reshape(batch_size, depth)
res1 = np.ones(shape=(batch_size, 1))
res0 = np.zeros(shape=(batch_size, 1))

for i in range(100001):
    noise1 = np.random.normal(0, 1, size=[batch_size, random_dim])
    noise2 = np.random.normal(0, 1, size=[batch_size, random_dim])
    noise3 = np.random.normal(0, 1, size=[batch_size, random_dim])
    generated_images1 = generator1.predict(noise1, verbose=False)  # вот генерация
    generated_images2 = generator2.predict(noise2, verbose=False)  # вот генерация
    generated_images3 = generator3.predict(noise3, verbose=False)  # вот генерация
    discriminator1.trainable = True  # хорошо, проверим
    discriminator2.trainable = True  # хорошо, проверим
    discriminator3.trainable = True  # хорошо, проверим
    discriminator1.train_on_batch(generated_images1, res0)  # вся ваша генерация - фейк!
    discriminator2.train_on_batch(generated_images2, res0)  # вся ваша генерация - фейк!
    discriminator3.train_on_batch(generated_images3, res0)  # вся ваша генерация - фейк!
    discriminator1.train_on_batch(image_batch1, res1)  # натуральные - не фейк
    discriminator2.train_on_batch(image_batch2, res1)  # натуральные - не фейк
    discriminator3.train_on_batch(image_batch3, res1)  # натуральные - не фейк

    discriminator1.trainable = False  # Проверил
    discriminator2.trainable = False  # Проверил
    discriminator3.trainable = False  # Проверил
    gan1.train_on_batch(noise1, res1)  # генератор учится подделывать генерацию
    gan2.train_on_batch(noise2, res1)  # генератор учится подделывать генерацию
    gan3.train_on_batch(noise3, res1)  # генератор учится подделывать генерацию
    if i%10 == 0:
        result1 = generator1.predict(np.expand_dims(noise1[0], axis=0)).reshape(height, width, 1)
        result2 = generator2.predict(np.expand_dims(noise1[0], axis=0)).reshape(height, width, 1)
        result3 = generator3.predict(np.expand_dims(noise1[0], axis=0)).reshape(height, width, 1)
        full = np.concatenate([result1, result2, result3], axis=2)
        print(full.shape)
        cv2.imwrite(filename=f"results/flowers_c/result{i}.jpg", img=full * 255)
    if i % 100 == 0:
        generator1.save(f"models/generatorRed{i}")
        generator2.save(f"models/generatorGreen{i}")
        generator3.save(f"models/generatorBlue{i}")
        discriminator1.save(f"models/discriminatorRed{i}")
        discriminator2.save(f"models/discriminatorGreen{i}")
        discriminator3.save(f"models/discriminatorBlue{i}")
