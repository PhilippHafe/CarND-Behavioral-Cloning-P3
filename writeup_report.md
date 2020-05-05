# **Behavioral Cloning**

## Writeup File

This writeup report summarizes the steps taken to solve the third assignment of the Udacity Nanodegree Programm

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/keras_model_summary.JPG "Model Summary"
[image2]: ./examples/LossHistory.jpg  "LossHistory"
[image3]: ./examples/center_2020_05_02_16_25_18_641.jpg "Center Driving"
[image4]: ./examples/Recovery1.jpg "Recovery Image"
[image5]: ./examples/Recovery2.jpg "Recovery Image"
[image6]: ./examples/Recovery3.jpg "Recovery Image"
[image7]: ./examples/Normal.jpg "Normal Image"
[image8]: ./examples/Flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing ``python drive.py model.h5``

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of five convolution layers with 5x5 filter sizes and a filter depth of 24 (model.py lines 78-82)
The convolutional layers include RELU activation functions layers to introduce nonlinearity.
To these layers, three fully connected layers are added (model.py lines 90-92). 
At the input of the network the data is normalized using a Keras lambda layer and cropped using a Cropping2D layer (code line 75-76). 

#### 2. Attempts to reduce overfitting in the model

Initially I choose LeNets architecture as a basis. To prevent overfitting the model, I introduced the following measures:
* I added pooling layers in addition to the convolutions
* I added two dropout layers after the convolutions
* I augmented the training data by flipping them horizontally 
* I extended the training data by using the images from the left and right camera as well

In my final model, the first two points are no longer implemented, since I chose a network similiar to NVIDIAs NN architecture presented [here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) .

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 22). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 95).

#### 4. Appropriate training data

The Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a LeNet architecture and then make the network more complex until the results looks satisfying.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it is very easy to implement and comparably cheap in training.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model and added dropouts and pooling layers as well as RELU activation functions.

With this architecture, I was even able to complete a whole round on track 1 without leaving the track. However the performance on the second track was horrible, so I needed the model to make better generalization.

I therefore switched the architecture to the one from NVIDIAs developers, using five convultional layers and discarding the pooling and dropout layers. This model architecture takes longer to train, but the losses are low on both the training and validation set.

The final step was to run the simulator to see how well the car was driving around track one. The vehicle does a pretty good job and stays centered in lane almost all the time. 
It is even able to generalize and drive for a couple of turns on the more challenging track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 75-92) consists of a convolution neural network with the layers and filter sizes shown in the above picture.
This picture was created using keras summary function: `model.summary()`

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to do this in autonomous mode in case it gets too close to the side.
These images show what a recovery looks like starting from leaving the track on the right:

![alt text][image4]
![alt text][image5]
![alt text][image6]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images horizontally thinking that this would help the network to generalize better, since the training data are biased towards left curves (negative steering angles).
For example, here is an image that has then been flipped:

![alt text][image7]
![alt text][image8]

After the collection process, I had 16779 number of data points. However, the driving results in autonomous mode looked not satisfying to me. I think, there are way more data points required to train the model. I therefore chose to use the recorded data provided by UDACITY.

I then preprocessed this data by normalizing the images and cropping them. These two steps are done as layers of the keras model (model.py line 76-77).

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the image above.
I used an adam optimizer so that manually training the learning rate wasn't necessary.

I trained for only three epochs since the losses did not increase significantly for more epochs. The history of losses is shown in the image below:

![alt text][image2]
