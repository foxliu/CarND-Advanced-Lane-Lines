
---

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

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test6.jpg "Road Transformed"
[image3]: ./examples/threshod_image.png "Binary Example"
[image4]: ./examples/warped_straight_lines.png "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[image7]: ./examples/test_image.png "Warp Example2"
[image8]: ./examples/test_undistort.png "Test Undistorted"
[image9]: ./examples/draw_data.png "Draw Lane and Data"
[video1]: ./project_video_output.mp4 "Project Video"
[video2]: ./challenge_video_output.mp4 "Challenge Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./model.ipynb" start (or in lines # 17 # of the file called `model.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image8]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]
Treatment effect
![alt text][image7]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 8 in `thresh.py`).  Here's some examples of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # 95 # in my code in `preprocess.py` in the funcetion `sliding_windows()` and lines 124 in the function `skip_sliding_windows()`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # 199 # in my code in `preprocess.py` in the function `draw_lane()` and for the convenience of viewing, I added a display on the top of the picture warped image and binary image int the lines # 241 # in my code in `preprocess.py` in the function `draw_data()`, Here is an example of my result on a test image:

![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's the [project video result](./project_video_output.mp4)

Here's the [challenge video result](./challenge_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

In the project I mainly use the R image Channel and S Channel of HLS image and their combination, sometimes the use of Abs Sobel X, the first time I tried to use one of them to complete, but in the face of shadows and low contrast surface effect is not good, the use of `cv2.blur` and `cv2.createCLAHE` are not much improved, by detecting the image that a lot of non lane region is detected, but there is a good effect in the other thresh binary image. I then use histogram to simply determine whether these images for which thresh binary by image is similar to `histogram = np.sum (channel, axis=1)` to show the height of the histogram by `len(histogram[histogram > 100]))` to decide whether to use the binary imge that still can't be improved effectively, and then try to put some of the image histogram the detected set as 0 methods, I implemented this step in lines in my code # 129 # in `preprocess.py` in the function `auto_choice_r_s_x()`, but the problem still exists.

To solve this problem by adding fit in my Line, I will first determine whether the fit is suitable for fit if not I will raise a Exception, I implemented this step in lines in my code in `line.py` in # # 36 the function (`add), In, lines 146 in `preprocess.py` in the # # function (`add_to_line) ` I call and to capture the exception, and then returned to the function `calc_curv_rad_and_center_dist (in), lines # 161 # in `preprocess.py` used to determine this and according to the programmed thresh_binar_image sequence in fun_names for processing. This makes the challenge_video results look much better. But harder_challenge_video.mp4 is still very bad.

I am thinking about whether I can find Lane in the form of convolution, or use deep learning to help determine what thresh to use to generate binary image

