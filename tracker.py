import cv2
import matplotlib.pyplot as plt
from math import hypot, sin, cos, atan2, pi
from errors import CaptureError, FrameError

class Axes(object):
    def __init__(self, width, height):
        self.origin = (width/2, height/2)
        self.angle = 0
        self.axisLength = hypot(width, height)
        self.dragCent = (width/2, height/4)

class Tracker(object):
    def __init__(self, videoPath=None, hsvLow=(0, 0, 0), hsvHigh=(255, 255, 255)):
        # Video capture parameters
        if videoPath is not None:
            self.capture = cv2.VideoCapture(videoPath)
            grabbed, frame = self.capture.read()
            if not grabbed:
                raise CaptureError("Failed to read video, file path may be incorrect.")
            self.height, self.width = frame.shape[:2]
        else:
            self.capture = cv2.VideoCapture()
            self.height = 0
            self.width = 0
        
        # Axes, parameter for mouse events, and final path frame
        self.axes = Axes(self.width, self.height)
        self.dragItem = 0
        self.pathFrame = None
        
        # HSV colourspace bounds
        self.hsvLow = hsvLow
        self.hsvHigh = hsvHigh
        
        # Scaling line parameters
        self.scaleLine = ((self.axes.origin[0] - 10, self.axes.origin[1] - 10), (self.axes.origin[0] - 10, self.axes.origin[1] - 50))
        self.scale = (100, "cm")
        
        # Positions on image
        self.imgPositions = []
        
        # Positions in re-defined coordinate system
        self.positions = []
        self.posTimes = []
        
        # Speed and component velocity values
        self.speeds = []
        self.xVel = []
        self.yVel = []
        self.velTimes = []
        
        # Acceleration and component acceleration values
        self.accel = []
        self.xAccel = []
        self.yAccel = []
        self.accelTimes = []
    
    def setVideo(self, videoPath):
        """Set the path to the video file"""
        self.release()
        self.capture = cv2.VideoCapture(videoPath)
        grabbed, frame = self.capture.read()
        self.height, self.width = frame.shape[:2]
        self.axes = Axes(self.width, self.height)
        self.scaleLine = ((self.axes.origin[0] - 10, self.axes.origin[1] - 10), (self.axes.origin[0] - 10, self.axes.origin[1] - 50))
        
    def setThresh(self, hsvLow, hsvHigh):
        """Set HSV colour threshold. Both parameters are passed as
        tuples of hue, saturation, and value."""
        for n in hsvLow:
            if n < 0 or n > 255:
                raise ValueError("HSV values must be between 0 and 255.")
        for n in hsvHigh:
            if n < 0 or n > 255:
                raise ValueError("HSV values must be between 0 and 255.")
        self.hsvLow = hsvLow
        self.hsvHigh = hsvHigh
    
    def drawAxes(self, frame):
        """Adds the axes defined by self.axes to the given frame."""
        self.axes.angle = atan2(self.axes.origin[0] - self.axes.dragCent[0], self.axes.origin[1] - self.axes.dragCent[1])
        s = sin(self.axes.angle)
        c = cos(self.axes.angle)
        l = self.axes.axisLength
        x = self.axes.origin[0]
        y = self.axes.origin[1]
        xAx1 = (int(-l*c + x), int(l*s + y))
        xAx2 = (int(l*c + x), int(-l*s + y))
        yAx1 = (int(-l*s + x), int(-l*c + y))
        yAx2 = (int(l*s + x), int(l*c + y))
        cv2.line(frame, xAx1, xAx2, (255, 0, 0), 2)
        cv2.line(frame, yAx1, yAx2, (51, 255, 255), 2)
        cv2.circle(frame, self.axes.dragCent, 4, (0, 0, 255), 2)
        cv2.circle(frame, self.axes.origin, 4, (0, 0, 255), 2)
        cv2.circle(frame, self.scaleLine[0], 4, (0, 0, 255), 2)
        cv2.circle(frame, self.scaleLine[1], 4, (0, 0, 255), 2)
        cv2.line(frame, self.scaleLine[0], self.scaleLine[1], (0, 252, 124), 2)
    
    def setupTrackbar(self, frameName, maxNum=200):
        def nothing(x):
            pass
        cv2.createTrackbar("Scale({})".format(self.scale[1]), frameName, 0, maxNum, nothing)
    
    def setScale(self):
        """Displays frame and allows user to manipulate the
        axes and scale line."""
        if self.pathFrame is None:
            raise FrameError("No frame available. Ensure track() has been called first.")
        else:
            cv2.namedWindow("Select scale")
            cv2.setMouseCallback("Select scale", self.eventHandler)
            self.setupTrackbar("Select scale")
            while True:
                mod = self.pathFrame.copy()
                self.drawAxes(mod)
                cv2.putText(mod, "Press c to continue", (self.width - 175, self.height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow("Select scale", mod)
                if cv2.waitKey(150) & 0xFF == ord("c"):
                    break
            self.scale = (cv2.getTrackbarPos("Scale({})".format(self.scale[1]), "Select scale"), "cm")
            cv2.destroyAllWindows()
    
    def track(self, outputName="", resizeWidth=0):
        """Tracks an object based on the HSV bounds supplied by the
        user. This function records position, velocity, acceleration,
        and time values, which are stored in the Tracker instance."""
        if self.capture is None:
            raise CaptureError("Video capture not initialized. Use setVideo(<videoPath>) to initialize.")
        
        if resizeWidth:
            self.height = int(float(resizeWidth)/self.width*self.height)
            self.width = int(resizeWidth)
            self.axes = Axes(self.width, self.height)
            self.scaleLine = ((self.axes.origin[0] - 50, self.axes.origin[1] - 50), (self.axes.origin[0] - 50, self.axes.origin[1] - 100))
            
        if outputName != "":
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            out = cv2.VideoWriter(outputName, fourcc, 30.0, (self.width, self.height))

        self.capture.set(1, 1)
        frameCount = 0
        while True:
            grabbed, frame = self.capture.read()
    
            if not grabbed:
                break

            if resizeWidth:
                frame = cv2.resize(frame, (resizeWidth, int(float(resizeWidth)/self.width*self.height)))
    
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.hsvLow, self.hsvHigh)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            conts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            center = None
            if len(conts) > 0:
                c = max(conts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                cv2.circle(frame, (int(x), int(y)), int(radius),(0, 252, 124), 1)
                cv2.circle(frame, center, 1, (0, 0, 255), 1)
                
                t = self.capture.get(0)/1000.0
                self.imgPositions.append(center)
                self.posTimes.append(t)
	
            for i in range(1, len(self.imgPositions)):
        		if self.imgPositions[i - 1] is None or self.imgPositions[i] is None:
        			continue
        		cv2.line(frame, self.imgPositions[i - 1], self.imgPositions[i], (0, 0, 255), 2)
            
            cv2.imshow("Tracker", frame)
            cv2.waitKey(1)
            
            self.pathFrame = frame.copy()
     
            if outputName != "":
                out.write(frame)
        if outputName != "":
            out.release()
        cv2.destroyAllWindows()
    
    def computeMotion(self):
        """Transforms coordinates to match the coordinate system
        defined by the axes and scale, then computes speed/velocity
        and acceleration. These are stored in the Tracker instance."""
        for x, y in self.imgPositions:
            c = cos(self.axes.angle)
            s = sin(self.axes.angle)
            scaleFactor = self.scale[0]/hypot(self.scaleLine[1][0] - self.scaleLine[0][0], self.scaleLine[1][1] - self.scaleLine[0][1])
            xNew = ((x - self.axes.origin[0])*c + (-y + self.axes.origin[1])*s)*scaleFactor
            yNew = (-(x - self.axes.origin[0])*s + (-y + self.axes.origin[1])*c)*scaleFactor
            self.positions.append((xNew, yNew))
        
        if len(self.positions) > 1:
            for i in range(1, len(self.positions)):
                p1 = self.positions[i - 1]
                p2 = self.positions[i]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                dt = self.posTimes[i] - self.posTimes[i - 1]
                tAvg = (self.posTimes[i] + self.posTimes[i - 1])/2.0
                s = hypot(dx, dy)/dt
                sx = dx/dt
                sy = dy/dt
                self.speeds.append(s)
                self.xVel.append(sx)
                self.yVel.append(sy)
                self.velTimes.append(tAvg)
        
        if len(self.speeds) > 1:
            for i in range(1, len(self.speeds)):
                dv = self.speeds[i] - self.speeds[i - 1]
                dvx = self.xVel[i] - self.xVel[i - 1]
                dvy = self.yVel[i] - self.yVel[i - 1]
                dt = self.velTimes[i] - self.velTimes[i - 1]
                tAvg = (self.velTimes[i] + self.velTimes[i - 1])/2.0
                self.accel.append(dv/dt)
                self.xAccel.append(dvx/dt)
                self.yAccel.append(dvy/dt)
                self.accelTimes.append(tAvg)
        
    def plotPos(self, comp=False):
        """Plot x-y position, irrespective of time. If comp is True,
        the x and y positions are plotted on separate subplots with
        respect to time."""
        if comp:
            f, axarr = plt.subplots(2, sharex=True, sharey=True, figsize=(12, 6))
            x = [p[0] for p in self.positions]
            y = [p[1] for p in self.positions]
            axarr[0].plot(self.posTimes, x, "k-", linewidth=1)
            axarr[1].plot(self.posTimes, y, "r-", linewidth=1)
            axarr[0].set_title("x Position")
            axarr[1].set_title("y Position")
            f.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor="none", top="off", bottom="off", left="off", right="off")
            plt.xlabel("Time(s)")
            plt.ylabel("Position({})".format(self.scale[1]), labelpad=15)
        else:
            plt.figure()
            x = [p[0] for p in self.positions]
            y = [p[1] for p in self.positions]
            plt.xlabel("x Position({})".format(self.scale[1]))
            plt.ylabel("y Position({})".format(self.scale[1]))
            plt.plot(x, y, "b-", linewidth=1)
        plt.show()
    
    def plotVel(self, comp=False):
        """Plot speed with respect to time. If comp is true, the component
        velocities are plotted on separate subplots."""
        if comp:
            f, axarr = plt.subplots(2, sharex=True, sharey=True, figsize=(12, 6))
            axarr[0].plot(self.velTimes, self.xVel, "k-", linewidth=1)
            axarr[1].plot(self.velTimes, self.yVel, "r-", linewidth=1)
            axarr[0].set_title("x Component of Velocity")
            axarr[1].set_title("y Component of Velocity")
            f.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor="none", top="off", bottom="off", left="off", right="off")
            plt.xlabel("Time(s)")
            plt.ylabel("Speed({}/s)".format(self.scale[1]), labelpad=15)
        else:
            plt.figure(figsize=(15, 5))
            plt.plot(self.velTimes, self.speeds, "k-", linewidth=1)
            plt.xlabel("Time(s)")
            plt.ylabel("Speed({}/s)".format(self.scale[1]))
        plt.show()
    
    def plotAcc(self, comp=False):
        """Plot scalar acceleration with respect to time. If comp is true,
        the component accelerations are plotted on separate subplots."""
        if comp:
            f, axarr = plt.subplots(2, sharex=True, sharey=True, figsize=(12, 6))
            axarr[0].plot(self.accelTimes, self.xAccel, "k-", linewidth=1)
            axarr[1].plot(self.accelTimes, self.yAccel, "r-", linewidth=1)
            axarr[0].set_title("x Component of Acceleration")
            axarr[1].set_title("y Component of Acceleration")
            f.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor="none", top="off", bottom="off", left="off", right="off")
            plt.xlabel("Time(s)")
            plt.ylabel("Acceleration({}/s^2)".format(self.scale[1]), labelpad=15)
        else:
            plt.figure(figsize=(15, 5))
            plt.plot(self.accelTimes, self.accel, "g-", linewidth=1)
            plt.xlabel("Time(s)")
            plt.ylabel("Acceleration({}/s^2)".format(self.scale[1]))
        plt.show()
    
    def toTxt(self, filename, outputType):
        """Writes motion quantities to a text file. outputType defines which
        quantities are written to the file:
            'p' - position, 'v' - speed/velocity, 'a' - acceleration"""
        with open(filename, "w") as f:
            if outputType == 'p':
                f.write("{:18}{:18}Time(s)\n".format("X({0})".format(self.scale[1]), "Y({0})".format(self.scale[1])))
                for i in range(0, len(self.positionss)):
                    f.write("{:<18.1f}{:<18.1f}{:<18.3f}\n".format(self.positions[i][0], self.positions[i][1], self.posTimes[i]))
            elif outputType == 'v':
                f.write("{:18}{:18}{:18}Time(s)\n".format("Speed({}/s)".format(self.scale[1]), "xVel({}/s)".format(self.scale[1]), "yVel({}/s)".format(self.scale[1])))
                for i in range(0, len(self.speeds)):
                    f.write("{:<18.1f}{:<18.1f}{:<18.1f}{:<18.3f}\n".format(self.speeds[i], self.xVel[i], self.yVel[i], self.velTimes[i]))
            elif outputType == 'a':
                f.write("{:18}{:18}{:18}Time(s)\n".format("Accel({}/s^2)".format(self.scale[1]), "xAccel({}/s^2)".format(self.scale[1]), "yAccel({}/s^2)".format(self.scale[1])))
                for i in range(0, len(self.accel)):
                    f.write("{:<18.1f}{:<18.1f}{:<18.1f}{:<18.3f}\n".format(self.accel[i], self.xAccel[i], self.yAccel[i], self.accelTimes[i]))
    
    def eventHandler(self, event, x, y, flags, param):
        """Function for handling mouse click and drag events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if hypot(x - self.axes.origin[0], y - self.axes.origin[1]) <= 5:
                self.dragItem = 1
                self.axes.dragCent = (self.axes.dragCent[0] + x - self.axes.origin[0], self.axes.dragCent[1] + y - self.axes.origin[1])
                self.axes.origin = (x, y)
            elif hypot(x - self.axes.dragCent[0], y - self.axes.dragCent[1]) <= 5:
                self.dragItem = 2
                self.axes.dragCent = (x, y)
            elif hypot(x - self.scaleLine[0][0], y - self.scaleLine[0][1]) <= 5:
                self.dragItem = 3
                self.scaleLine = ((x, y), self.scaleLine[1])
            elif hypot(x - self.scaleLine[1][0], y - self.scaleLine[1][1]) <= 5:
                self.dragItem = 4
                self.scaleLine = (self.scaleLine[0], (x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragItem:
                if self.dragItem == 1:
                    self.axes.dragCent = (self.axes.dragCent[0] + x - self.axes.origin[0], self.axes.dragCent[1] + y - self.axes.origin[1])
                    self.axes.origin = (x, y)
                elif self.dragItem == 2:
                    self.axes.dragCent = (x, y)
                elif self.dragItem == 3:
                    self.scaleLine = ((x, y), self.scaleLine[1])
                elif self.dragItem == 4:
                    self.scaleLine = (self.scaleLine[0], (x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragItem = 0
    
    def release(self):
        self.capture.release()
    
    