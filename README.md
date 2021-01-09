## Project2 : Advanced Lane Finding
---

[//]: # (Image References)

[image1]: ./output_images/undistorted_img.png "Undistorted"
[image2]: ./output_images/undistorted_img2.png "Road Transformed"
[image3]: ./output_images/binary_thresholds.png "Binary Example"
[image4]: ./output_images/perspective_transform.png "Warp Example"
[image5]: ./output_images/finding_lanes.png "Fit Visual"
[image6]: ./output_images/result.png  "Output"


**The goals / steps of this project are the following:**

**Camera Calibration:**
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

**Pipeline:**
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

**The python file name is : project2.ipynb**

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image

For camera calibration, the main step is using chessboard images to obtain image points and object points. and then undistorting image using cv2.undistort() function.


The object points will be the (x,y,z) coordinates of the chessboard corners in the world. Assuming the chessboard is fixed on the (x,y) plane at z=0, the object points are the same for each calibration image. The image points will be appended with the (x,y) pixel position of each of the corners in the image plane with each successful chessboard detection.


Use the output variables to apply distortion correction. And then can obtain the undistorted image result, which is displayed below.


![alt text][image1]

### Pipeline (test images)

#### 1. Provide an example of a distortion-corrected image.

The distortion correction image that was calculated via camera calibration is presented here.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of binary thresholds applying the `Sobel operator` in the x-direction on the original image and `color threshold` using the S channel (HLS). The output is shown below.

the function name is `camera_image_processing()` in the code.


![alt text][image3]

(Left) stacked image:
 <span style="color:green">the green</span> is `the gradient threshold component` and <span style="color:blue">the blue</span> is `the color channel threshold component`.

 (Right) black and white combined thresholded image:
 this one has combined both gradient and color threholds into one image.


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform can change view like a bird's eye on the road. This is useful for calculating the lane curvature. I assume the camera is mounted at the center of the car, such that the lane center is the midpoint at the bottom of the image between the two lines the car has detected. The offset of the lane center from the center of the image (converted from pixels to meters) is a distance from the center of the lane.

```python
def camera_perspective_transform(img):
    img_size = (img.shape[1], img.shape[0])
    center = [img_size[0]/2, img_size[1]/2]
    offset1 = [210,200]
    offset2 = [360,300]
    offset3 = [13, 7]

    #plt.plot(center[0], center[1],'^')
    #plt.plot(center[0]-offset1[0]+offset3[1], center[1]+offset1[1],'.')
    #plt.plot(center[0]+offset1[0]+offset3[1], center[1]+offset1[1],'.')
    #plt.plot(center[0]-offset2[0]+offset3[0], center[1]+offset2[1],'.')
    #plt.plot(center[0]+offset2[0]+offset3[0], center[1]+offset2[1],'.')

    src = np.float32([[center[0]-offset1[0]+offset3[1], center[1]+offset1[1]], [center[0]+offset1[0]+offset3[1], center[1]+offset1[1]],
                      [center[0]-offset2[0]+offset3[0],center[1]+offset2[1]], [center[0]+offset2[0]+offset3[0],center[1]+offset2[1]]])
    dst = np.float32([[center[0]-offset2[0]+offset3[0], center[1]+offset1[1]], [center[0]+offset2[0]+offset3[0], center[1]+offset1[1]],
                      [center[0]-offset2[0]+offset3[0],center[1]+offset2[1]], [center[0]+offset2[0]+offset3[0],center[1]+offset2[1]]])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    inv_M = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    return warped, M, inv_M
```

| Source        | Destination|
|:-------------:|:----------:|
| 437, 560      | 293, 560   |
| 857, 560      | 1013, 560  |
| 293, 660      | 293, 660   |
| 1013, 660     | 1013, 660  |

![alt text][image4]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The first step is to implement Sliding Windows and Fit a polynomial. Then it can derive two polynomial functions. The second step is to find lane lines using exist polynomial functions. From now on, searching the lanes around the polynomial function can skip the first step.

the function name is `camera_lane_searching()` in the code.

```python
# HYPERPARAMETER
margin = 100

# Grab activated pixels
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

leftx1, lefty1, rightx1, righty1, out_img = find_lane_pixels(binary_warped)

left_fit = np.polyfit(lefty1, leftx1, 2)
right_fit = np.polyfit(righty1, rightx1, 2)

left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                left_fit[1]*nonzeroy + left_fit[2] + margin)))
right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                right_fit[1]*nonzeroy + right_fit[2] + margin)))

# Again, extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

# Fit new polynomials
left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

```

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To derive a radius of curvature on the road, it needs to measuring x,y. We can estimate using a warped image. To get the length. First, we need to know the pixels of the warped image.

| x [pixel]     | y [pixel]  |
|:-------------:|:----------:|
| 290           | 680        |
| 1010          | 680        |
| 290           | 0          |
| 1010          | 0          |

![alt text][image4]

Consider lane width of 3.7 meters and the projected image length is 30 meters.  So the x value is 30/720, and y value is 30/680.


```python
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/680 # meters per pixel in y dimension
xm_per_pix = 3.7/720 # meters per pixel in x dimension

```

We can get curvature using a second-order polynomial curve of fitted left and right lanes. Also, the car's center position can calculate using
mean of the constant coefficient of left/right second-order polynomial curve.

the function name is `camera_lane_searching()` in the code.
```python
# get curvature
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

y_eval = np.max(ploty)

left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

position = (left_fit_cr[2] + right_fit_cr[2])/2
distance_from_center = (binary_warped.shape[1]*xm_per_pix)/2 - position
```


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Overall pipelines are shown below.

```python
def camera_lane_detection(img):  
    #[mtx, dist] = camera_calibation()
    dst_img = cv2.undistort(img, mtx, dist, None, mtx)
    [result1, result2] = camera_image_processing(dst_img)
    [warped_img, matrix, inv_matrix] = camera_perspective_transform(result2)
    [result,left_curvature, right_curvature, distance_from_center] = camera_lane_searching(warped_img)
    newwarp = cv2.warpPerspective(result, inv_matrix, (result.shape[1], result.shape[0]))
    img = cv2.addWeighted(dst_img, 1, newwarp, 0.3, 0)
    cv2.putText(img, "Distance from center : %f [m]" % (distance_from_center), (50, 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, (0, 0, 0))
    cv2.putText(img, "L_K: %f, R_K: %f" % (left_curvature , right_curvature), (50, 200), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, (0, 0, 0))
    return img
```
![alt text][image6]
---


### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images//project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The algorithm must detect two lanes. However, sometimes lanes are not clear or vanished. It needs to develop to calculate center position despite detecting just one lane for robust behavior. This algorithm will not good or not working when there exist lots of shadow or obstacles. The weakest thing is road surface marking. The algorithm will not work properly.
