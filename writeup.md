# **Behavioral Cloning Project**


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[center_image]: ./examples/center_2016_12_01_13_31_14_194.jpg
[left_image]: ./examples/left_2016_12_01_13_31_14_194.jpg
[right_image]: ./examples/right_2016_12_01_13_31_14_194.jpg

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

------------------------------------------------------------------------------

### Model Architecture and Training Strategy

#### 1. Appropriate Training Set & Training Process

##### 1.1 Collecting Training Data
Training data was chosen to keep the vehicle driving on the road. I used the sample training data provided by Udacity as it contains already more that 24,000 images that with the appropriate preprocessing and augmenting could be enough to train the model and achieve the desired results.

The recorded data contains couple of laps using center lane driving. Here are examples of the center, left and the right camera images:

![alt text][center_image]

![alt text][left_image]

![alt text][right_image]

##### 1.2 Correcting steering angle for left and right images
I used not only the center images to train the model but also the left and right images.
I added a small correction to the steering angle taken by the left and right cameras to help the car steer back to the road if it approaches a steep turn.

##### 1.3 Augmenting the training data
To augment the data set, I also flipped images and angles thinking that this would generalize the model even more. This gives the model more training data.

This also helps balancing the training data and preventing the model from being biased towards steering into a specific direction.

##### 1.4 Data Preprocessing
I then preprocessed this data by normalizing it between -0.5 and 0.5 in addition to clipping the top and bottom parts as they don't really contribute in describing the road that the car is driving on.

##### 1.5 Validation Data
I used this training data for training the model. The validation set (20% of the collected data) helped determine if the model was over or under fitting.

#### 2. Solution Design Approach

##### 2.1 Which Model to choose?
The overall strategy was to derive the simplest possible model architecture and test it and then improve the model till reaching the desired behavior.

I started by testing a simple model with just a Flatten layer then a Dense layer but this was too simple to solve this rather complicated regression problem.

I then tested the LeNet architecture with couple of convolutional layer, pooling layers, flatten layer then three Dense layers. This approach yields better results but then model was

##### 2.2 Final Model Architecture
Finally I choose to model the Nividia architecture with the following layers:

- Normalize the data between -0.5 and 0.5. to have an average mean.
- Add three convolutional layers with 5x5 kernel and 24, 36 and 48 feature maps respectively.
- Add two convolutional layers with 3x3 kernel and 64 feature maps.
- Flatten layer
- Three Dense layer with output size 100, 50 and 10 respectively.
- Dropout layers after the convolutional layers and after each one of the previous dense layers.
-Finally, a Dense layer with one output to give the predicted steering angle.

The model includes ReLU as an activation function for the convolutional and Dense layers to introduce nonlinearity.

##### 2.3 Overfitting
To combat the overfitting, as I modified in the model description above, I added three dropout layers after the convolutional layers and after each one of the previous dense layers.
I choose 3 as the number of epochs as I found out that more that this and the model is overfitting.

##### 2.4 Autonomous Driving
Also the I decreased the learning rate of the Adam optimizer to be 0.001 to make the model slowly  approach the desired training loss without fluctuating.

At the end of the process, the vehicle is able to drive autonomously around the first track without leaving the road.
