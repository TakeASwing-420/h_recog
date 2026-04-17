import string
import matplotlib
matplotlib.use("Agg")

import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
from imutils import build_montages

from pyimagesearch.az_dataset import load_mnist_dataset
from pyimagesearch.az_dataset import load_az_dataset
from pyimagesearch.models import ResNet

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--az", required=True, help="path to input AZ dataset")
ap.add_argument("-m", "--model", required=True, type=str, help="path to output model")
ap.add_argument("-p", "--plot", required=True, type=str, help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

EPOCHS = 50
INIT_LR = 1e-1
BS = 128

print("[INFO] loading AZ dataset...")
azData, azLabels = load_az_dataset(args["az"])

print("[INFO] loading MNIST dataset...")
digitsData, digitsLabels = load_mnist_dataset()

azLabels = azLabels + 10

data = np.vstack([azData, digitsData])
labels = np.hstack([azLabels, digitsLabels])   # keep integer labels here

(trainX, testX, trainLabels, testLabels) = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels,
)

numClasses = len(np.unique(labels))
classTotals = np.bincount(trainLabels, minlength=numClasses).astype("float32")
safeTotals = np.where(classTotals == 0, 1.0, classTotals)
classWeight = safeTotals.max() / safeTotals
classWeight = {i: float(weight) for i, weight in enumerate(classWeight)}

data_augmentation = Sequential([
    layers.RandomRotation(10/360),
    layers.RandomZoom(0.05),
    layers.RandomTranslation(0.1, 0.1),
])

print(numClasses)
print("[INFO] data and labels prepared for training...")

opt = SGD(learning_rate=INIT_LR, momentum=0.9)
model = ResNet.build(32, 32, 1, numClasses, (3, 4, 6), (64, 64, 128, 256), reg=0.0005)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

def preprocess(image, label, training=False):
    image = tf.expand_dims(image, axis=-1)
    image = tf.image.resize(image, (32, 32))
    image = tf.cast(image, tf.float32) / 255.0
    if training:
        image = data_augmentation(image, training=True)
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((trainX, trainLabels))
train_ds = train_ds.shuffle(buffer_size=10000)
train_ds = train_ds.map(
    lambda x, y: preprocess(x, y, True),
    num_parallel_calls=tf.data.AUTOTUNE,
)
train_ds = train_ds.batch(BS)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((testX, testLabels))
val_ds = val_ds.map(lambda x, y: preprocess(x, y, False), num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BS).prefetch(tf.data.AUTOTUNE)

H = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=classWeight,
    verbose=1,
)

labelNames = list(string.digits + string.ascii_uppercase)

print("[INFO] evaluating network...")
predictions = model.predict(val_ds)
print(classification_report(testLabels, predictions.argmax(axis=1), target_names=labelNames))

print("[INFO] serializing network...")
model.save(args["model"], save_format="h5")

# construct a plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc ="lower left")
plt.savefig(args["plot"])

images = []

for i in np.random.choice(np.arange(0, len(testLabels)), size=(49,)):
    sample = tf.expand_dims(testX[i], axis=-1)
    sample = tf.image.resize(sample, (32, 32))
    sample = tf.cast(sample, tf.float32) / 255.0

    probs = model.predict(sample[np.newaxis, ...], verbose=0)
    prediction = probs.argmax(axis=1)
    label = labelNames[prediction[0]]

    image = testX[i].astype("uint8")
    color = (0, 255, 0)

    if prediction[0] != testLabels[i]:
        color = (0, 0, 255)

    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    images.append(image)

montage = build_montages(images, (96, 96), (7, 7))[0]
cv2.imshow("OCR Results", montage)
cv2.waitKey(0)
