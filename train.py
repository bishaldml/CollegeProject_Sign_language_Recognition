from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


# Step 1 - Building the CNN
classifier = Sequential()

classifier.add(Conv2D(32 , (3,3) ,padding='same', activation = 'relu', input_shape = (128,128,3)))
classifier.add(MaxPool2D((2,2)))

classifier.add(Conv2D(64 , (3,3) ,padding='same',  activation = 'relu'))
classifier.add(MaxPool2D((2,2)))
classifier.add(Dropout(0.3))

classifier.add(Conv2D(128 , (3,3) ,padding='same', activation = 'relu'))
classifier.add(MaxPool2D((2,2)))
classifier.add(Dropout(0.3))

classifier.add(Conv2D(128 , (3,3) ,padding='same', activation = 'relu'))
classifier.add(MaxPool2D((2,2)))
classifier.add(Dropout(0.3))

classifier.add(Flatten())

classifier.add(Dense(units = 512 , activation = 'relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(units = 17 , activation = 'softmax')) 

classifier.summary()


# Compiling the CNN
classifier.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy']) 
# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)



# Step 2 - Preparing the train/test data and training the model

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False,
   )

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(128, 128),
                                                 batch_size=10,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(128, 128),
                                            batch_size=10,
                                            class_mode='categorical') 

# history = classifier.fit(
#       training_set,
#       steps_per_epoch=training_set.samples/training_set.batch_size,
#       epochs=10,
#       validation_data= test_set,
#       validation_steps=test_set.samples/test_set.batch_size,
#       callbacks=[early_stopping])
history = classifier.fit(
      training_set,
      steps_per_epoch=len(training_set)/training_set.batch_size,
      epochs=15,
      validation_data=test_set,
      validation_split=0.2,
      #validation_steps=len(test_set)/test_set.batch_size,
      callbacks=[early_stopping])


# Saving the model
model_json = classifier.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(model_json) #model saved
classifier.save_weights('model2.h5')     #Weights saved



# Graph:
import matplotlib.pyplot as plt
# plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# plot the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


