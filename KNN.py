import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Set up the image data generator for training and testing data
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.15,
                                   zoom_range=0.15,
                                   horizontal_flip=False)
test_datagen = ImageDataGenerator(rescale=1./255)

# Set up the training and testing data generators
training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(128, 128),
                                                 batch_size=32,
                                                 class_mode='categorical')
test_set = test_datagen.flow_from_directory('data/test1',
                                             target_size=(128, 128),
                                             batch_size=32,
                                             class_mode='categorical')



# for k in range(1, 100):
    # Create a KNN classifier with k=1-100
knn = KNeighborsClassifier(n_neighbors=4)

# Train the classifier on the training set
X_train, y_train = training_set.next()
X_train = np.reshape(X_train, (len(X_train), -1))
y_train = np.argmax(y_train, axis=1)
for i in range(len(training_set)-1):
    X, y = training_set.next()
    X = np.reshape(X, (len(X), -1))
    y = np.argmax(y, axis=1)
    X_train = np.vstack((X_train, X))
    y_train = np.hstack((y_train, y))
knn.fit(X_train, y_train)
# Test the classifier on the testing set
X_test, y_test = test_set.next()
X_test = np.reshape(X_test, (len(X_test), -1))
y_test = np.argmax(y_test, axis=1)
for i in range(len(test_set)-1):
    X, y = test_set.next()
    X = np.reshape(X, (len(X), -1))
    y = np.argmax(y, axis=1)
    X_test = np.vstack((X_test, X))
    y_test = np.hstack((y_test, y))
acc = knn.score(X_test, y_test)
print('Accuracy:',acc)
Class_labels = ['0','1','2','3','4','5','A','B','C','D','E','F','G','H','I','J','Null']
print (classification_report(y_test, knn.predict(X_test), target_names=Class_labels))

