#import csv
import pandas as pd
import numpy as np
import cv2
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Lambda, Dropout, Activation, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
import math

#Read the recorded data with pandas
data = pd.read_csv('./data/driving_log.csv', header=0, names=['CenterCamera','LeftCamera','RightCamera','Steering','Throttle','Brake','Speed'])
#Replace the /root/Desktop Path in the image path columns with the path needed here
#data.replace(to_replace='/root/Desktop', value = '.', regex=True, inplace=True)
data.replace(to_replace=' IMG/', value = './data/IMG/', regex=True, inplace=True)
data.replace(to_replace='IMG/', value = './data/IMG/', regex=True, inplace=True)

#Split training and validation data
train_samples, validation_samples = train_test_split(data, test_size = 0.2)

# Define the generator
def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        #Loop through the samples and get btach samples
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            #Get the steering measurments
            #measurements = data['Steering'].tolist()
            measurements = []
            STEERING_CORRECTION = 0.4
            #Get the images from the image paths
            images = []
            for i in batch_samples.index: #Loop through all the image paths of the current batch
                #Load the center, left and right camera images
                img_center = cv2.cvtColor(cv2.imread(batch_samples['CenterCamera'][i]), cv2.COLOR_BGR2RGB)
                img_left = cv2.cvtColor(cv2.imread(batch_samples['LeftCamera'][i]), cv2.COLOR_BGR2RGB)
                img_right = cv2.cvtColor(cv2.imread(batch_samples['RightCamera'][i]), cv2.COLOR_BGR2RGB)
                steering = batch_samples['Steering'][i]
                steering_left = steering + STEERING_CORRECTION
                steering_right = steering - STEERING_CORRECTION

                #Append tthe images and steering angles to the list
                images.extend([img_center, img_left, img_right])
                measurements.extend([steering, steering_left, steering_right])

            #Augment training data by flipping images
            augmented_images, augmented_measurements = [],[]
            for image,measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)

            #Convert to np arrays for training the model
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            
            #Yield argument for generator
            yield sklearn.utils.shuffle(X_train, y_train)
            
#Commands for using the generator
BATCH_SIZE = 32

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)


# Implement keras model
model = Sequential()
model.add(Lambda(lambda x: x / 255.5 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,(5,5), activation='relu'))
model.add(Conv2D(24,(5,5),activation='relu'))
model.add(Conv2D(24,(5,5),activation='relu'))
model.add(Conv2D(24,(5,5),activation='relu'))
model.add(Conv2D(24,(5,5),activation='relu'))
#model.add(Conv2D(6,kernel_size=(5, 5),activation='relu'))
#model.add(MaxPooling2D())
#model.add(Dropout(0.5))
#model.add(Conv2D(6,kernel_size=(5, 5),activation='relu'))
#model.add(MaxPooling2D())
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

# Run keras model 
model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/BATCH_SIZE), validation_data = validation_generator, validation_steps=math.ceil(len(validation_samples)/BATCH_SIZE), epochs=3, verbose=1)

model.save('model.h5')


#Plot history
plot = True
if plot == True:
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    #plt.show()
    plt.savefig('./figure.jpg')


