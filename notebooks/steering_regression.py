# 1) Data loading and preprocessing
import os, csv, cv2, numpy as np
from sklearn.model_selection import train_test_split

base = '../data/udacity'

def load_udacity(base):
    samples = []
    with open(os.path.join(base, 'driving_log.csv')) as f:
        reader = csv.reader(f)
        for row in reader:
            img_path = os.path.join(base, 'IMG', os.path.basename(row[0]))
            angle = float(row[3])
            samples.append((img_path, angle))
    return samples

def preprocess(img):
    img = img[60:135, :, :]            # crop sky/hood
    img = cv2.resize(img, (200,66))    # NVIDIA PilotNet size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = img / 255.0
    return img

samples = load_udacity(base)
train, val = train_test_split(samples, test_size=0.2, random_state=42)

def generator(samples, batch_size=64):
    n = len(samples)
    while True:
        np.random.shuffle(samples)
        for offset in range(0, n, batch_size):
            batch = samples[offset:offset+batch_size]
            X, y = [], []
            for pth, ang in batch:
                img = cv2.imread(pth)
                if img is None: continue
                X.append(preprocess(img)); y.append(ang)
            yield np.array(X), np.array(y)


# 2) Model and training
import tensorflow as tf
from tensorflow.keras import layers, models

def pilotnet():
    m = models.Sequential([
        layers.Input(shape=(66,200,3)),
        layers.Conv2D(24, (5,5), strides=(2,2), activation='relu'),
        layers.Conv2D(36, (5,5), strides=(2,2), activation='relu'),
        layers.Conv2D(48, (5,5), strides=(2,2), activation='relu'),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(1)
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')
    return m

model = pilotnet()
train_gen = generator(train)
val_gen = generator(val)

steps_tr = max(1, len(train)//64)
steps_va = max(1, len(val)//64)

history = model.fit(train_gen, steps_per_epoch=steps_tr,
                    validation_data=val_gen, validation_steps=steps_va,
                    epochs=10)

os.makedirs('../models/steering', exist_ok=True)
model.save('../models/steering/pilotnet_udacity.h5')
