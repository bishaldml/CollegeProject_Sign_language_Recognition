from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint


# Step 1 - Building the CNN
classifier = Sequential()
classifier.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (128,128,1)))
classifier.add(BatchNormalization())
classifier.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

classifier.add(Conv2D( 64, (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(BatchNormalization())
classifier.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

classifier.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

classifier.add(Flatten())

classifier.add(Dense(units = 128 , activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 26 , activation = 'softmax')) 





## Initializing the CNN
#classifier = Sequential()
#
## First convolution layer and pooling
#classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
#classifier.add(MaxPooling2D(pool_size=(2, 2)))
##classifier.add(Dropout(0.3))
#
## Second convolution layer and pooling
#classifier.add(Convolution2D(64, (3, 3), activation='relu'))
#classifier.add(MaxPooling2D(pool_size=(2, 2)))
##classifier.add(Dropout(0.3))
#
#classifier.add(Convolution2D(128, (3, 3), activation='relu'))
#classifier.add(MaxPooling2D(pool_size=(2, 2)))
#
## Flattening the layers
#classifier.add(Flatten())
#
## Adding a fully connected layer
#classifier.add(Dense(units=64, activation='relu'))
##classifier.add(Dropout(0.3))
#classifier.add(Dense(units=6, activation='softmax')) # softmax for more than 2 classes
#
#
#early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')
#best_model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, save_weights_only=False, monitor='val_accuracy', mode='max', verbose=1)

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2
classifier.summary()



# Step 2 - Preparing the train/test data and training the model

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        shear_range=0.2,
        zoom_range=0.2)
       # horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(128, 128),
                                                 batch_size=1,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(128, 128),
                                            batch_size=1,
                                            color_mode='grayscale',
                                            class_mode='categorical') 
history = classifier.fit(
        training_set,
        steps_per_epoch=2600, # No of images in training set
        epochs=20,
        validation_data=test_set,
        validation_steps=780)  # No of images in #test set
        #callbacks=[early_stop, best_model_checkpoint])

# evaluating the model
_, acc = classifier.evaluate_generator(training_set, steps=2600,
                                  verbose=0)
print('Test Accuracy: %.3f' % (acc * 100))


# Saving the model
model_json = classifier.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json) #model saved
classifier.save_weights('model1.h5')     #Weights saved



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





