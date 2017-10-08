
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Files

**main files:**

- [./utils.py](./utils.py): utility functions file
- [./line_finding.py](./line_finding.py): main class LineFinding file
- [./Advanced_Lane_Finding.ipynb](./Advanced_Lane_Finding.ipynb): tutorial notebook
- [./project_output.mp4](./project_output.mp4): output video

**output_image files:**

- [./output_images/camera_cal_undistort](./output_images/camera_cal_undistort): undistort chessboard images
- [./output_images/gradient_pic](./output_images/gradient_pic): gradient map of test images
- [./output_images/perspective_trans](./output_images/perspective_trans): perspective transformed test images
- [./output_images/finding_lanes](./output_images/finding_lanes): perspective transformed lanes with fitted lines
- [./output_images/imposing_lanes](./output_images/imposing_lanes): test images with fitted lines
- [./output_images/pipe_line](./output_images/pipe_line): final transformed images


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 7 through 30  of the file called `utils.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

**original image:**

![这里写图片描述](http://img.blog.csdn.net/20171008203151072?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

**undistort image:**

![这里写图片描述](http://img.blog.csdn.net/20171008203123542?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![这里写图片描述](http://img.blog.csdn.net/20171008203431417?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 32 through 86 in `utils.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![这里写图片描述](http://img.blog.csdn.net/20171008203528962?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_trans()`, which appears in lines 88 through 92 in the file `utils.py` . The `perspective_trans()` function takes as inputs source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[577,463],[707,463],[244,688],[1059,688]])
dst = np.float32([[244,300],[1059,300], [244,700],[1059,700]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 577, 463      | 244,300        | 
| 707, 463      | 1059,300    |
| 244, 688     | 244,700     |
| 1059, 688      | 1059,700        |

I verified that my perspective transform was working as expected by drawing transformed image to verify that the lines appear parallel in the warped image.

![这里写图片描述](http://img.blog.csdn.net/20171008204132105?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial  which appears in lines 94 through 228 in the file `utils.py`  kinda like this:

![这里写图片描述](http://img.blog.csdn.net/20171008204225671?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 230 through 244 in my code in `utils.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 275 through 299 in my code in `utils.py` in the function `impose_lane()`.  Here is an example of my result on a test image:

![这里写图片描述](http://img.blog.csdn.net/20171008204459909?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWW91bmdfR3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The highlight lane region isn't stable in some cases. I think the reason is that the region we found isn't perfect. 

To make it more robust, I think I should fine tune parameters so that on the one hand the region can be correct, on the other hand the region change can be smooth.