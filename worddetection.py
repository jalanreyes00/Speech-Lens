from picamera import PiCamera
from picamera.array import PiRGBArray
from imutils.object_detection import non_max_suppression
import pyttsx3
import cv2
import numpy as np
import pytesseract
import time
import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library

is_video = True
take_photo = False

# GPIO set up
GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
GPIO.setup(8, GPIO.IN,pull_up_down=GPIO.PUD_DOWN) #Switch position 2
GPIO.setup(10, GPIO.IN,pull_up_down=GPIO.PUD_DOWN) #Switch position 1
GPIO.setup(12, GPIO.IN,pull_up_down=GPIO.PUD_DOWN) #Button

def button_callback(channel):
    global is_video
    global take_photo
    if not is_video:
        #capture picture for scanning
        take_photo = True

def switch_callback(channel):    
    global is_video
    #video capture mode off
    is_video = False
    
def switch2_callback(channel):    
    global is_video
    global take_photo
    #video capture mode on
    is_video = True
    
GPIO.setwarnings(False) # Ignore warning for now
GPIO.add_event_detect(12,GPIO.RISING,callback=button_callback, bouncetime=200) # Setup event on pin 12 rising edge
GPIO.add_event_detect(10,GPIO.RISING,callback=switch_callback, bouncetime=200) # Setup event on pin 10 rising edge
GPIO.add_event_detect(8,GPIO.RISING,callback=switch2_callback, bouncetime=200) # Setup event on pin 8 rising edge

#--DETECT/RECOG CODE--
    
def getX(val):
    return val[1]

def detect_words(image, rW, rH):
    # Read image
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (320,320)), 1.0, (320, 320), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    (rects, confidences) = decode_predictions(scores, geometry)
    rects = non_max_suppression(np.array(rects), probs=confidences)
    i = 0
    results = []

    for rect in rects:
        newrect = rect * np.array([rW,rH,rW,rH])
        x0 = int(rect[0] * rW)
        y0 = int(rect[1] * rH)
        x1 = int(rect[2] * rW)
        y1 = int(rect[3] * rH)
        dx = int((x1 - x0) * 0.1)
        dy = int((y1 - y0) * 0.1)    
        y0 = max(0, y0-dy)
        y1 = min(y1+dy*2, h)
        
        # Creates box around text
        roi = image[y0:y1, x0:x1]
        
        text = pytesseract.image_to_string(roi, config=config)
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        results.append([text, x0, y0])
        cv2.rectangle(image, (x0,y0), (x1,y1), (0,0,255), 2)
        
        # Creates text for box
        label = "{}: {}".format("TEXT", text)
        cv2.putText(image, label, (x0,y0), font, 0.5, (0,0,255), 2)
        i = i + 1
        
    cv2.imshow("Output", image)  
    results.sort(key=getX)
    for word in results:
        engine.say(word[0])
        engine.runAndWait()

def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < 0.9:
                continue
            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)

# Resolution and buffer constants
X_RESOLUTION = 640
Y_RESOLUTION = 480
BUFFER = X_RESOLUTION * 0.15

#Camera Setup
camera = PiCamera()
camera.resolution = (X_RESOLUTION,Y_RESOLUTION)
camera.framerate = 30
video_feed = PiRGBArray(camera, size=(X_RESOLUTION,Y_RESOLUTION))
video_feed.truncate(0)
video_feed.seek(0)

# Set up text to speech engine
engine = pyttsx3.init()

# CNN layers needed (scores and geometry of rectangles for detected words)
layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]

# Config for tesseract
config = ("-l eng --oem 1 --psm 7")

# Calculate frequency of system
fr_calc = 1
freq = cv2.getTickFrequency()

# Font
font = cv2.FONT_HERSHEY_SIMPLEX

# Read cnn model
net = cv2.dnn.readNet("frozen_east_text_detection.pb")


# Coefficient to transform new coordinates to fit image
rW = float(X_RESOLUTION / 320)
rH = float(Y_RESOLUTION / 320)

# Camera warm-up
time.sleep(0.1)

#--CONTROL LOOP--
while True:
    if is_video:
        for frame in camera.capture_continuous(video_feed, format="bgr", use_video_port=True):
            if not is_video:
                break
            t1 = cv2.getTickCount()
            
            # Take image from camera
            image = frame.array
            
            # Detect words and speak
            detect_words(image, rW, rH)
            
            t2 = cv2.getTickCount()
            
            # Calculate FPS
            del_t = (t2-t1)/freq
            fr_calc = 1/del_t
            
            video_feed.truncate(0)
            video_feed.seek(0)
            
            key =  cv2.waitKey(1)
            if key == ord('q'):
                break
    else:
        if take_photo:
            time.sleep(5)
            camera.capture('image.jpg')
            image = cv2.imread('image.jpg')
            # Detect words and speak
            detect_words(image, rW, rH)
            take_photo = False
cv2.destroyAllWindows()