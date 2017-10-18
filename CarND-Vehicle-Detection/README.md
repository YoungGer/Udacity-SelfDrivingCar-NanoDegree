
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.



## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

## Files

**main files:**

- [./utils_detection.py](./utils.py): utility functions file
- [./Vehicle-Detection.ipynb](./Advanced_Lane_Finding.ipynb): tutorial notebook
- [./project_output.mp4](./project_output.mp4): output video

**output_image files:**

- [./output_images/vehicle.png](./output_images/vehicle.png): a vehicle image
- [./output_images/ycrcb-hog.png](./output_images/ycrcb-hog.png): hog image of ycrcb channels of the vehicle image
- [./output_images/search_windows.png](./output_images/search_windows.png): search windows
- [./output_images/original_box.png](./output_images/original_box.png): search window output box
- [./output_images/pipeline.png](./output_images/pipeline.png): pipelines of the test images

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 10 through 26 of the file called `utils_detection.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![这里写图片描述](http://img.blog.csdn.net/20171018083942163?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20171018084002926?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![这里写图片描述](http://img.blog.csdn.net/20171018085214907?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and choose the one which balances classifier accuracy and speed.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the combination of spatial features, hist color features and hog features. First I split data into training and testing sets. I used training data and grid search to find best parameters (C=0.01) which was accord with highest cross validation accuracy. Then I did inference on test data sets using the best model and got `99.04%` accuracy which I thought was pretty good. To get final model, I used best parameters and all data to retrain the model and got the final model which was saved as `model.pkl`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I tried many parameters and manually selected best parameters which performed well on test images and video streams.

| Scale | Y_start | Y_end | Reason |
| ------------- |-------------| -----|-----|
| 0.8 | 400 | 500 |Find Far Cars|
| 1.5 | 400 | 656 |Find All Cars|




#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![这里写图片描述](http://img.blog.csdn.net/20171018092509634?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20171018092824603?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Finally, I added rules to filter some false positive bounding boxes. The rule is contained in lines 200 through 201 in `utils_detection.py`. The rules are:

- width>50
- height>50 
- 0.33 < width/height < 2.85



Here's an example result of test images:



![这里写图片描述](http://img.blog.csdn.net/20171018111524146?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Problems:

- When two cars met, the bounding box contained two cars.
- The bounding box was sometimes too big while others too small.
- There  were some false positive especially in another road and shadows.

Future Improvements:

- The quality of classifier is critical. So I need to try other models (CNN for example) and more data.
- The object region is implemented using simple sliding windows. Maybe I can try other methods, like RCNN, Faster-RCNN, etc.


