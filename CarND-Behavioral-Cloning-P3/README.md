# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 30-34) 

The model includes RELU layers to introduce nonlinearity (code lines 30-34), and the data is normalized in the model using a Keras lambda layer (code line 29). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 35). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 52-57). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 50).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use convolution neural network.

My first step was to use a convolution neural network model similar to the [The NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). I thought this model might be appropriate because it succeeds in NVIDIA's experiments.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added Dropout layer so that each node could have an importance in model training.

Then I trained my model again and kept training as long as the validation loss was decreasing.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I collected path recovering data at these spots.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 28-40) consisted of a convolution neural network with the following layers and layer sizes:

- Image normalization
- Convolution: 5x5, filter: 24, strides: 2x2, activation: RELU
- Convolution: 5x5, filter: 36, strides: 2x2, activation: RELU
- Convolution: 5x5, filter: 48, strides: 2x2, activation: RELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
- Drop out (0.5)
- Fully connected: neurons: 100, activation: RELU
- Fully connected: neurons:  50, activation: RELU
- Fully connected: neurons:  10, activation: RELU
- Fully connected: neurons:   1 (output)


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:


![center.jpg](http://upload-images.jianshu.io/upload_images/2255998-7f8d480b09449e09.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to go back to road center when it was off the center. These images show what a recovery looks like starting from left side to center:


![left1.jpg](http://upload-images.jianshu.io/upload_images/2255998-5b6e8f6234db421a.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![left2.jpg](http://upload-images.jianshu.io/upload_images/2255998-b0ac784444e2dedf.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![left3.jpg](http://upload-images.jianshu.io/upload_images/2255998-29442e8ef3f39ee2.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would augment data and decrease always-left-turn bias. For example, here is an image that has then been flipped:


![center.jpg](http://upload-images.jianshu.io/upload_images/2255998-6749c63e710a1b39.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![center 2.jpg](http://upload-images.jianshu.io/upload_images/2255998-eb410de3e80d8d51.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



After the collection process, I had 24580 number of data points. I then preprocessed this data by normalization and cropping to interest region.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by training and validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
