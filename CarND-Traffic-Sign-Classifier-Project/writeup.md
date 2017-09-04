# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the `len` function to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here are the exploratory visualizations of the data set.

The picture below describes what it's like in each class.

![这里写图片描述](http://img.blog.csdn.net/20170904104647844?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

The picture below describes the distribution of classes in the training, validation and test set.

![这里写图片描述](http://img.blog.csdn.net/20170904104933186?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the original RGB image contains too much information which is not too much related to the classes. At the same time, I can use less memory, reduce model complexity, speed up model calculations.

Here is an example of a traffic sign image before and after grayscaling.

![这里写图片描述](http://img.blog.csdn.net/20170904110123330?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
![这里写图片描述](http://img.blog.csdn.net/20170904110222467?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

As a last step, I normalized the image data because normalization makes different features have similar distributions so that training can be faster and easier.

I decided to generate additional data because deep learning needs a lot of data. However, The number of some classes data is too small and we cannot train an efficient classifier without enough data.

To add more data to the the data set, I used rotation, affine transform, translations.

Here is an example of an original image and an augmented image:

![这里写图片描述](http://img.blog.csdn.net/20170904110123330?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
![这里写图片描述](http://img.blog.csdn.net/20170904110748117?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


The difference between the original data set and the augmented data set is the following:

- rotation
- affine transform
- translations


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GRAY image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Droupout					|	keep_prob=0.7											|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flat	    | outputs 400      									|
| Fully connected		| outputs 120    									|
| RELU					|												|
| Droupout					|	keep_prob=0.7											|
| Fully connected		| outputs 84    									|
| RELU					|												|
| Droupout					|	keep_prob=0.7											|
| Fully connected		| outputs 43    									|
| Softmax				| etc.        									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

- Optimizer: Adam
- Batch_size: 128
- Epochs: 60
- Learning Rate: 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 96.6%
* validation set accuracy of 94.2%
* test set accuracy of 92.3%

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?
I first tried LeNet. Because the last tutorial has provided LeNet code.
* What were some problems with the initial architecture?
First, the input image has only one channel while the input of LeNet has three channels. Second, it easily overfits.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I add dropout layer to solve overfitting problem.
* Which parameters were tuned? How were they adjusted and why?
I tuned learning rate. When learning rate is small, the speed of study is slow. When learning rate is large, it may not study effectively. I tuned learning rate according to learning curve.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
The convolution layer is important because it has less  parameters than fully connected layer so that the model complexity is small. At the same time, the parameters are shared across different picture batches so that kernel can detect same pattern in different regions.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![这里写图片描述](http://img.blog.csdn.net/20170904141426662?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

The first image might be difficult to classify because the pattern is difficult to recognize when the picture is resized to 32*32.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bicycles crossing      		| Slippery road  									| 
| General caution     			| General caution  										|
| Roundabout mandatory				| Roundabout mandatory										|
| Turn left ahead	      		| Turn left ahead					 				|
| Speed limit (60km/h)			| Speed limit (60km/h)	     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 92.3%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 28th cell of the Ipython notebook.

For the first image, the model predicts a children crossing sign with probability of 0.86 while the probability of true label bicycles crossing is 12.5%

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .86         			| Children crossing									| 
| .13     				| Bicycles crossing										|
| .01					| Beware of ice/snow											|
| .00	      			| Ahead only				 				|
| .00				    | Slippery Road      							|


For the second image, the model is relatively sure that this is a general caution (probability of 1.0), and the image does contain a general caution sign.

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Children crossing									| 
| .00     				| Road narrows on the right										|
| .00					| Pedestrians											|
| .00	      			| Keep right				 				|
| .00				    | Wild animals crossing     							|

For the third image, the model is relatively sure that this is a roundabout mandatory (probability of 1.0), and the image does contain a roundabout mandatory sign.

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| Roundabout mandatory								| 
| .00     				| Traffic signals									|
| .00					| Beware of ice/snow											|
| .00	      			| Right-of-way at the next intersection				 				|
| .00				    | Turn left ahead     							|

For the fourth image, the model is relatively sure that this is a turn left ahead (probability of 1.0), and the image does contain a turn left ahead sign.

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Turn left ahead								| 
| .00     				| Ahead only							|
| .00					| Yield										|
| .00	      			| Keep right				 				|
| .00				    | No passing   							|

For the fifth image, the model is totally wrong. The true label is speed limit (60km/h) while the model predicts speed limit (50km/h) with 99% probability. The reason may lies in that the two signs are too similar.

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| Speed limit (50km/h)								| 
| .00     				| Speed limit (60km/h)							|
| .00					| Go straight or left									|
| .00	      			| Speed limit (30km/h)				 				|
| .00				    | Keep left							|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Original picture:

![这里写图片描述](http://img.blog.csdn.net/20170904143133276?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

conv1 feature map:

![这里写图片描述](http://img.blog.csdn.net/20170904143153634?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

conv2 feature map:

![这里写图片描述](http://img.blog.csdn.net/20170904143212863?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

The first convolution layer extracts the triangle character. The second convolution layer extracts the edge character. 


