# Speech-Lens
Code and model used to create Speech Lens for DubHacks 2020
frozen_east_text_detection.pb was the model used for text detection. 
worddetection.py is the python script that runs the Speech Lens algorithm.

# Inspiration
We wanted to utilize hardware in our hack and create a tangible product that could improve lives. Our goal was to create a offline low-cost wearable solution to empower the blind and visually impaired. Our device helps identify words in sight by converting text to spoken speech. We hope to provide a non-contact alternative to braille that can be accessed without an internet connection. 

# What it does
The Speech Lens has two settings: photo and video mode. In photo mode, the user presses a button to take a photo and text in view is spoken from left-to-right. In video mode, the camera feed is constantly processed and detected words are verbally communicated. These modes can be toggled using a switch. Speech Lens interprets these images using EAST, a deep neural network that identifies and draws regions of interest (ROI) around text. These regions are processed using Tesseract, an optical character recognition engine. Lastly, the pyttsx3 text-to-speech library is used to read the text aloud to the user.

# How we built it
We used a Raspberry Pi 3 with Pi Camera v2.1 for initial prototyping and development, and the final prototype was created using a Raspberry Pi Zero W. The device is powered using a external 10000mAh battery bank. We used the  the GPIO pins to implement a button for photo capture and a slide switch to change operational modes. The main script was completely written in Python and the openCV/Tesseract/EAST open source was found online.


# Helpful Installation links for OpenCV and Tesseract
https://heartbeat.fritz.ai/real-time-object-detection-on-raspberry-pi-using-opencv-dnn-98827255fa60
https://stackoverflow.com/questions/53347759/importerror-libcblas-so-3-cannot-open-shared-object-file-no-such-file-or-dire (For us, one dependency was not able to install, but opencv still worked)
