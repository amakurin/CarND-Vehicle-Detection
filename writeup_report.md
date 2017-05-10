**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

My project includes the following files:
* labutils.py containing routines to extract fatures from images, build and train classifier, slide window, classify window etc. 
* main.py containing the Pipeline class and script to start video processing
* project_result.mp4 containing resulting video produced by running pipeline on file `project_video.mp4`
* writeup_report.md this file summarizing the results


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 80 through 131 of `labutils.py`. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(6, 6)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I wrote a couple of routines to try different HOG parameters, parameters of histogram of colors and spatial features, and different combinations of feature types. The code for this step is contained in lines 357 through 425 of `labutils.py`. Function `brute_force_params` generates grid-like combinations of parameters, and for each combination performs training of classifier, and evaluation of accuracy and speed of feature extraction and prediction.

I ran this routine few times, first with wide ranges of parameters (about 30000 combinations) and low number (1000) of samples, to exclude most inefficient combinations, then narrowed ranges (about 7500 combinations) and increased number of samples to 2000, and finally ran it on full dataset with about 40 combinations of parameters.

It turned out that most time consuming part of prediction was feature extraction, with HOG extraction on first place. Next most time consuming operation (color histogram computation) was order of magnitude less.

But in the same time including HOG features in feature vector lead to increasing accuracy.

Unfortunately i didn't found any combination of parameters with acceptable accuracy with computing HOG on less then two channels.

For two channels the best choise was L and S channel of HLS.
For three channels the best choise of color space was YCrCb.

I didn't see any dependency between chosen color space and other HOG parameters. Actually no matter what color space and channels were chosen the following combination of parameters always gave best accuracy:

| Parameter        | Value         | 
|:----------------:|:-------------:| 
| orientations     |      9        | 
| pixels_per_cell  |      6        |
| cells_per_block  |      2        |

When i implemented HOG precalculation algorithm, i found out that even precalculation takes most of the processing time and  gives nothing more then 2 frames per second on my laptop, which is far away from real time.

I found some ways to speed up HOG calculation, which are of two main classes: use some lib with ready to use detector, like [dlib](http://dlib.net/train_object_detector.py.html), or go [low level](http://www.phyxs.com/ihog) with integral images, integral histograms and lookup tables. Both approaches are out of scope of this project, so i decided to concern more about accuracy and robustness.

So i chose 3-channelled HOG on YCrCb color space.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Analysis of different combinations of parameters and differen feature types showed that top accuracy of linear SVM is achieved with including all three explored types of features: color histograms of R, G and B channels, spatially binned RGB images of size 32x32, and 3-channelled HOG on YCrCb color space.

I wrote few routines to extract features of these types: `spatial_feats` lines 45-53 of `labutils.py`; `hist_feats` lines 55-78 of `labutils.py` and `hog_feats` lines 105-131 of `labutils.py`.

Then i trained linear SVM with this set of features 285-341 of `labutils.py` 

I used mixture of GTI vehicle database and KITTI database of images as source for samples of [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [nonvehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) classes.

I got 8792 vehicle and 8968 nonvehicle images, total of 17760. All images of size 64x64 pixels.

I prepared samples from these images by extracting features mentioned above and concatenating them into one flattened vector for each image.

I normalized all feature vectors by removing the mean and scaling.

Before training i randomly shuffled samples and reserved 20% as test set. As a result i got training set of size 14208 and test set of size 3552.

I got accuracy of 0.9935 on test set.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search vehicles only on the bottom half of the frames - region of the road. I used 4 different windows to compensate perspective distortion of vehicle size. After project video analysis i decided that window height of 48 pixels will be smallest, as distance from camera to these vehicles are big enough (20-25 meters) to consider it safe.

After some experiments with different window widths and overlaps i came to following 4 sliding window specifications:

| Search range in X | Search range in Y | Window size X,Y | Window overlap in X,Y| 
|:-----------------:|:-----------------:|:----------------:|:--------------:| 
|   0,1261          |     400,470       |     80,48        | 0.7,0.7        | 
|   0,1261          |     400,482       |     80,64        | 0.8,0.7        |
|   0,1261          |     410,526       |     132,96       | 0.8,0.5        |
|   0,1261          |     420,680       |     128,128      | 0.5,0.5        |

For each window specification i calculated scale ratios in x and y directions of frame, then i scaled frame and slide standart 64x64 window. I chose this implementation to make HOG precalculation possible.

The code for computing windows is contained in lines 427 through 479 of `labutils.py`. 

The code for whole search algorithm is in lines 504-568 `labutils.py`. 

Here are examples of all 4 window specifications

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Despite high accuracy on test set, classifier gave to much false positives on test images and video frames. 

I tried decision function thresholding without much luck. It really can be used to filter out false positives but in some very important cases thresholding will misclassify vehicles as well, which is not acceptable.

So i decided to go with 'hard negative mining' technique. I collectected all images that corresponded to false positive predictions on project video, added them to `nonvehicles` dataset and retrained classifier.

I got accuracy 0.9909 on test set, but much less false positives on project video frames. 

Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

I also collected heatmaps for 3 last frames and peformed thresholding on integrated heatmap to better filter out false positives.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are three frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all three frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

##### Computational complexity

The main problem i faced was computational complexity of feature extraction.
Actually my current implementation is not ready to use for real time processing, but i see few directions to beat this. 

One obvious way is to implement one of fast HOG algorithms, this includes precalculation of integral images and integral histograms, as well as lookup tables to reduce number of computations. After that one can consider methods to reduce number of windows to search. Actually on project video one could ignore leftmost part of frame because of separation rail.  

Another way is to get rid of HOG and any particular feature extraction and go with deep learning, using CNN as a classifier. I'm not sure that this way is suitable for modest processors but with powerfull GPU, i guess, this is the way to go.  

##### Accuracy

As i mentioned above despite high accuracy on test set, classifier predicted to much false positives before hard negative mining was applied. Even after that there are still number of false positives that potentially dangerous (eg in the middle of empty lane). 
My guess, that CNN could work better here as well. But more intelligent tracking could be another option.
