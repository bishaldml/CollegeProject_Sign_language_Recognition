import numpy as np
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the saved model
from keras.models import model_from_json

json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load the saved weights
loaded_model.load_weights("model1.h5")

# Load the test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(128, 128),
                                            class_mode='categorical',
                                            shuffle=False) 

Class_labels = ['0','1','2','3','4','5','A','B','C','D','E','F','G','H','I','J','NUll']

# Make predictions on the test set
y_pred = loaded_model.predict(test_set)

# Convert the predicted probabilities to class labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Get the true labels
y_true_labels = test_set.classes

# Compute the confusion matrix
confusion_mtx = confusion_matrix(y_true_labels, y_pred_labels)

# Plot the confusion matrix as a heatmap
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='viridis', xticklabels=Class_labels, yticklabels=Class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
# Get classification report
class_report = classification_report(y_true_labels, y_pred_labels, target_names=Class_labels)
print("Classification report:")
print(class_report)
