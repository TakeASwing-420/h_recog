import os
import numpy as np

os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.datasets import mnist

def load_mnist_dataset():
    # load the MNIST dataset and stack the training data and testing data
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
    data = np.vstack([trainData, testData])
    labels = np.hstack([trainLabels, testLabels])
    # return a tuple of the data and labels
    return (data, labels)

def load_az_dataset(azPath):
    # initialize the list of data and labels
    data = []
    labels = []
    # loop over the rows of the A-Z dataset CSV file
    with open(azPath) as f:
        for row in f:
            row = row.strip()
            if not row:
                continue
            # parse the label and image from the row, then update the
            # data and labels lists, respectively
            parts = row.split(",")
            label = int(parts[0])
            image = np.array(parts[1:], dtype="uint8")
            image = image.reshape((28, 28))
            data.append(image)
            labels.append(label)
    # convert the data and labels to NumPy arrays, then return a tuple
    # of the data and labels
    return (np.array(data, dtype="uint8"), np.array(labels, dtype="int32"))