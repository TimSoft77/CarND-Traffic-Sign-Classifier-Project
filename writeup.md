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

[image1]: ./writeup_images/unpreprocessed_image.png "Original image"
[image2]: ./writeup_images/preprocessed_image_y_channel.png "Preprocessed Image Y Channel"
[image8]: ./writeup_images/probabilities.PNG
[image3]: ./web_traffic_signs/no entry (16).jpg "LKJSKD"
[image4]: ./web_traffic_signs/pedestrians (27).jpg
[image5]: ./web_traffic_signs/road work (25).jpg
[image6]: ./web_traffic_signs/speed limit 30 (1).jpg
[image7]: ./web_traffic_signs/stop (14).jpg

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the python and numpy libraries to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43
* All the features and labels sets (e.g., X_train, y_valid) were numpy arrays with dtype=uint8 (thus each entry was an integer between 0 and 255 inclusive)

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is an image of a random trafic sign. I wanted to get an image of a traffic sign at this point to better evaluate the impact of pre-processing techniques.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

The [published paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) suggested converting to YUV, and normalizing the Y channel.  I converted to YUV with OpenCV, and applied its histogram equalization function to the Y channel (I do not believe this is identical to what is suggested in the published paper).  [Wikipedia's article on histogram equalization](https://en.wikipedia.org/wiki/Histogram_equalization) provides an example of the technique that suggests it might help make images clearer.  The enhanced Y channel of the exploratory visualization shown below supports that claim.

![alt text][image2]

The Udacity Self Driving Car Eng course suggested normalizing every value in each image with the equation "new_value = (old_value - 128)/128", so I did that too.

I wrote a function that only converted the images to grayscale and normalized, but ultimately did not use it.

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I made no changes to the training, validation, and testing sets beyond what was done in the first cell. In cell 7 I shuffled the training data.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in cell 4.

I used the LeNet architecture with 2 notable changes:
1. Adaptations for 3 color channels in the input images and 43 output classes
2. Dropout (this made a huge difference to the results)

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in cells 5, 6, and 8.

I trained the model in the exact same way as in the LeNet lab.  I could find no good reason to change the optimizer.

I did experiment with L2 regularization to reduce over-fitting, but it was nowhere near as effective as dropout, and was not included in the final training algorithm.

Number of epochs was left at 10.  Given that the validation accuracy was highest in the final epoch, and that the difference between it and training accuracy was similar through the last few epochs, I do not believe it should have been changed.

I never tried adjusting learning rate or batch size.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the cells 8 and 14.

My final model results were:
* training set accuracy of 0.991
* validation set accuracy of 0.941
* test set accuracy of 0.925

My process for finding a solution went like this:
1. Get a LeNet architecture working on the data.  This over-fit and was not accurate enough.
2. Try out normalization and preprocessing techniques as described above.  I initially did them manually (i.e., coded RGB to YUV conversion algorithms found in Wikipedia), but then switched to using OpenCV.  These improved accuracy, but there was still severe over-fitting, and accuracy was still below 0.9.
3. Changed the model to use more features in the conv layers, but only one fully connected layer.  This improved accuracy to about 0.91, but there was still severe over-fitting.
4. Implemented L2 Regularization.  It was ineffective.
5. Switched back to regular LeNet, but implemented dropout.  Great results. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web.  I manually (i.e., through Windows Photos app) cropped out a square region around the images of the signs.  Upon loading them in Ipython, I used OpenCV to resize them to 32x32 and convert from BGR to RGB.

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7]

I expected the stop and pedestrians images to be easy to classify.  The no entry image is at an angle, and the other two images have a lot of noise in their backgrounds, which may have made them difficult.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the cell 11.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road Work     		| Road Work   									| 
| No Entry    			| No Entry 								|
| Pedestrians					| Road Narrows on the Right											|
| 30 km/h	      		| 30 km/h						 				|
| Stop			| Stop     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 92.5%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 13th cell of the Ipython notebook.

The probabilities for each of the image predictions are plotted below.

![alt text][image8]

The classifier had near-complete confidence (and was correct) in its predictions for 3 of the images.  It was least confident in its prediction for the image it got wrong (pedestrians), and the third most probable prediction was correct (albeit with only a 0.026% probability).
