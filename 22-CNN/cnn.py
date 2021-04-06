# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:13:49 2021

@author: axvargas
"""
# Convolutional Neural Network

# =============================================================================
# PART 1 Build the CNN
# =============================================================================
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
#Init the ANN
classifier = Sequential()

# =============================================================================
# # STEP 1 CONVOLUTION
# =============================================================================

## NOTE: filters = 32, kernel_size = (3, 3)  -> 32 mapas de caractiristicas con ventanas de 3*3
## NOTE: activation = 'relu', bc we need to eliminate the linearity in the images
classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
classifier.add(BatchNormalization())

# =============================================================================
# # STEP 2 MAX POOLING
# =============================================================================
## NOTE: pool_size = (2,2) el tama;o de la ventana del pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Dropout(0.25))

# =============================================================================
# REPEAT THE PROCCESS
# =============================================================================

classifier.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Dropout(0.25))


classifier.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Dropout(0.25))



# =============================================================================
# # STEP 3 FLATTENING
# =============================================================================
classifier.add(Flatten())

# =============================================================================
# # STEP 4 ENTERING THE CNN
# =============================================================================

classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#compile the ANN
## NOTE: En caso de tener mas categorias usar otra "loss" = otra
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# =============================================================================
# PART 2 BRING THE IMAGES
# =============================================================================
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

testing_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

history = classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=testing_set,
        validation_steps=2000)

##8000 and  2000 is better


# =============================================================================
# SHOW GRAPHIC OF THE CURVES
# =============================================================================
import matplotlib.pyplot as plt
# plot loss
plt.subplot(211)
plt.title('Cross Entropy Loss')
plt.plot(history.history['loss'], color='blue', label='train')
plt.plot(history.history['val_loss'], color='orange', label='test')
# plot accuracy
plt.subplot(212)
plt.title('Classification Accuracy')
plt.plot(history.history['accuracy'], color='blue', label='train')
plt.plot(history.history['val_accuracy'], color='orange', label='test')
    
#show plot
plt.tight_layout()
plt.show()
# =============================================================================
# SAVE AND LOAD A MODEL
# =============================================================================
classifier.save('my_classifier')
saved_classifier = load_model('my_classifier')



# =============================================================================
# PREDICTIVE IMAGES
# =============================================================================
from os import walk
import numpy as np 
_, _, filenames = next(walk('dataset/test_image'))

list_of_imgs = []
for filename in filenames:
    image = load_img('dataset/test_image/' + filename, target_size=(128,128))
    arr = img_to_array(image)
    list_of_imgs.append(arr)


input_arr = np.array(list_of_imgs)  # Convert single image to a batch.
predictions = saved_classifier.predict(input_arr)
