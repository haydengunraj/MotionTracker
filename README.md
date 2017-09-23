# MotionTracker

Based on the (messy) tracking code contained in [daphTrack](https://github.com/haydengunraj/daphTrack), I decided to make a neater, more generalized toolset for tracking and analyzing the motion of objects. MotionTracker allows for object tracking using HSV thresholding. Moreover, a coordinate system and scale can be set on the video using simple drag and drop commands in order to convert image coordinates into real-world spatial coordinates. As a result, position, velocity, and acceleration values for the object are computed in terms of real-world dimensions, allowing for meaningful analysis of the tracker data.

![Track](https://github.com/haydengunraj/MotionAnalysis/blob/master/samples/tracking.gif "Tracking")

## Example Usage

```python
# import Tracker
from MotionTracker.tracker import Tracker

# Create Tracker instance with parameters
t = Tracker(videoPath="inputName.mov", hsvLow=(39, 0, 27), hsvHigh=(130, 255, 74))

# Alternatively:
#   t = Tracker()
#   t.setVideo("inputName.mov")
#   t.setThresh((39, 0, 27), (130, 255, 74))

# Track, with the image width being reset to 700p
# Notably, output needs to be a .avi to work properly
t.track(outputName="outputName.avi", resizeWidth=700)
```

At this point, a video will be shown depicting the tracking of the object, as shown in the GIF above.

Next, we need to set a coordinate system and scale for our tracker.

```python
# Set axes and scale for the image
t.setScale()
```

Here, the final frame of the tracking image will be shown, as well as axes and a scaling line (see below). The axes and scaling line operate on drag-and-drop controls at the red circles. The y-axis is yellow and the x-axis is red, and the control point on the y-axis is used to rotate the axes. After moving the scaling line, the real-world length of the scaling line can be set using the trackbar.

![Scale](https://github.com/haydengunraj/MotionAnalysis/blob/master/samples/axes.png "Scaling")

Notably, the example above is sideways, but can still be analysed in a 'normal' coordinate system by changing the axes.

We can new release our video capture, compute values which represent the object's motion, and plot these values.

```python
# Release video capture
t.release()

# Compute motion values
t.computeMotion()

# Plot positions
t.plotPos()
t.plotPos(comp=True)

# Plot speed and velocity components
t.plotVel()
t.plotVel(comp=True)

# Plot acceleration and its components
t.plotAcc()
t.plotAcc(comp=True)
```

The plotting functions above display position, velocity, and acceleration values in terms of real-world quantities. Some samples taken from the ball drop example are shown below.

![Pos](https://github.com/haydengunraj/MotionAnalysis/blob/master/samples/position.png "Position")
![Acc](https://github.com/haydengunraj/MotionAnalysis/blob/master/samples/acceleration.png "Acceleration")

### Dependencies

- [Python 2.7](https://www.python.org/downloads/)
- [OpenCV](http://opencv.org/)
- [MatPlotLib](https://matplotlib.org/)
