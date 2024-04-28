# CSL7360: Computer Vision

# Project: Air Writing and Detection

# Team Members

| Name          | Roll No  |
| ------------- | -------- |
| Navneet Kumar | B21CS050 |
| Ayush Jain    | B21BB006 |
| Soham Parikh  | B21CS074 |
| Het Dave      | B21CS020 |

## How To Run The Code

There are two ipynb files in the repository. One is for training of our digit Recogniton Model and the other is for the Air Writing and Detection, which utilizes this trained model.

The trained model is saved in the file `model.h5`. This file is used in the `Air_Writing_and_Detection.ipynb` file.

You need to install libraries, which can be done by running the following command in the terminal:

``pip install -r requirements.txt``at

To run the code you can click run all cells in the `Air_Writing_and_Detection.ipynb` file. This will open the webcam and you can start writing in the air. The code will detect the digits you write in the air and display them on the screen.

To retrain or make changes to the model or dataset, you can run the `Digit_Recognition.ipynb` file. This will train the model and save it in the file `model.h5`. This file can then be used in the `Air_Writing_and_Detection.ipynb` file.

## Problem Statement

After coronavirus outbreak,we had a couple of changes in our daily lifstlye to get used with.
We got to know the importance of online plateforms however for colaboration and learning purposes we need to have a whiteboard to write and explain things to others.

In today's digital age, with the increasing demand for remote collaboration and interactive learning, there is a growing need for innovative solutions that enable users to annotate and interact with virtual content seamlessly. One such application is a Virtual Whiteboard system that leverages computer vision techniques to detect fingers and allows users to write or draw on a digital canvas in real-time using their fingertips.

The goal of this project is to develop a robust Virtual Whiteboard system that utilizes a webcam to detect finger movements accurately and translate them into digital ink on a virtual canvas. This system should provide an intuitive and interactive interface for users to write, draw, and interact with the content without the need for physical input devices such as a stylus or mouse.

Key Objectives:

- **Real-time Finger Detection**: Implement computer vision algorithms to detect and track finger movements in real-time using the webcam feed. The system should accurately recognize the position, orientation, and movement of fingers to enable precise interaction.
- **Virtual Canvas Creation**: Develop a virtual canvas where users can write and draw using their fingers.
- **Gesture Recognition**: Incorporate gesture recognition algorithms to interpret different finger gestures, such as using index finger to write on the virtuall canvas, or using the thumb to end the session.
- **Digit Recognition**: Train a machine learning model to recognize digits written by the user on the virtual canvas and convert it into digital numbers for further processing.

# Methodology

The methodology for the Air Writing and Detection project involves several key steps, leveraging computer vision techniques and machine learning models to enable users to write or draw on a digital canvas in real-time using their fingertips. The process can be broadly divided into the following sections:

## 1. Real-time Finger Detection

The first step in the methodology is the real-time detection and tracking of fingers. This is achieved using the MediaPipe library, which provides a robust solution for hand and finger tracking. The `mpHands` module from MediaPipe is utilized to detect hands in each frame of the video feed. For each detected hand, the landmarks of the fingers are extracted, which are then used to determine the position and orientation of the fingers in the frame.
However ,this sounds trivial but it was not that easy to implement as it sounds. The main challenge was to detect the different type of gestures to track the fingers and then to write the digits in the air we had to make sure that the digits are written in a proper way so that the model can recognize it.
This is carried out by the `HandDetector` class which has the following methods:

- `__init__`: Initializes the `HandDetector` object with the necessary parameters for hand detection.
- `find_hands`: Detects hands in the input frame using the MediaPipe library.
- `find_position`: Extracts the landmarks of the fingers from the detected hands.
- `find_distance`: Calculates the Euclidean distance between two points in the frame.
- `find_angle`: Calculates the angle between three points in the frame.
- `find_direction`: Determines the direction of the fingers based on the landmarks.
- `find_fingers_up`: Determines which fingers are up based on the landmarks.

This is done in the following steps:

- **Step 1**: Capture the video feed from the webcam.
- **Step 2**: Process each frame of the video feed to detect hands. They are converted from BGR to RGB and resized and normalized for efficient processing.
- **Step 3**: Extract the landmarks of the fingers from the detected hands. These landmarks are used to determine the position and orientation of the fingers in the frame. This utilizes a CNN model trained on the MediaPipe dataset to detect the landmarks accurately.
- **Step 4**: Analyze the landmarks to determine the state of the fingers (open or closed) and the position of the index finger. This information is used for gesture recognition and digit recognition.
- **Step 5**: Display the detected fingers and gestures on the screen in real-time to provide feedback to the user.

## 2. Virtual Canvas Creation

Once the fingers are detected, a virtual canvas is created on which the user can write or draw. This is done by mapping the detected finger positions to a 2D space, effectively creating a digital canvas. The canvas is dynamically updated as the user moves their fingers, allowing for real-time interaction.
For this to work we have created a `Writer` class which has the following methods:

- `__init__`: Initializes the `Writer` object with an empty list of points, a Z-distance(based on size of hand) ,threshold, a direction, and a writing threshold.
- `calibrate`: Sets the Z-distance, threshold, direction, and writing threshold.
- `on_canvas`: Checks if a point is on the canvas based on the Z-distance and direction.
- `write`: Adds a point to the list of points if the point is on the canvas.
- `writeV2`: Adds a point to the list of points if the point is on the canvas and the distance to the last point is less than the writing threshold.
- `writeV3`: Adds a point to the list of points if the point is on the canvas and the time gap between this point and the last point is less than 0.5.
- `clear`: Clears the list of points.
- `draw`: Draws lines between consecutive points in the list of points on the given image.
- `range_points`: Returns the minimum and maximum x and y coordinates among the points in the list.
  every time the user points on the screen if the point is on the canvas then the point is added to the list of points and then the line is drawn between the points to show the user that the digit is being written on the screen.
  A new writer object is created everytime when the user stops pointing and then starts pointing again to write a new digit.
  This is done to make sure that the digits are written properly for the digit recognition model to recognize it. So,we can seperate the digits written in the air.

Else all points by all writers can be drawn on the screen and then we can use clustering to seperate the digits written in the air.

## 3. Gesture Recognition

To interpret the user's gestures, the system checks the state of the fingers (open or closed) and the position of the index finger. This is done using custom functions that analyze the landmarks of the fingers. For instance, if the index finger is pointing upwards and the other fingers are closed, it could be interpreted as a specific gesture (e.g., starting a new line).

## 4. Digit Recognition

After the user has written a digit on the virtual canvas, the system needs to recognize and interpret this digit. This is achieved by extracting the image of the digit from the canvas and preprocessing it to match the input requirements of the digit recognition model. The preprocessed image is then passed through a Convolutional Neural Network (CNN) model, which has been trained to recognize handwritten digits. The model outputs the recognized digit, which is then displayed on the screen.

## 5. Model Training and Evaluation

The digit recognition model is trained using a dataset of handwritten digits. The model is evaluated using metrics such as accuracy, precision, recall, and F1 score to ensure its effectiveness in recognizing digits written in the air. The model's performance is continuously monitored and improved upon based on the evaluation results.

## 6. Integration and User Interface

Finally, the various components of the system are integrated to provide a seamless user experience. The user interface allows the user to interact with the system easily, with real-time feedback on the detected gestures and recognized digits. The system is designed to be intuitive, enabling users to write or draw on the virtual canvas using their fingertips without the need for physical input devices.

This methodology combines computer vision techniques for real-time finger tracking and gesture recognition with machine learning for digit recognition, providing a novel solution for air writing and detection.

## Observations and Conclusions

Step 1:Air writing

<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images\result_image1.png" alt="Logo" height="480" width="640">
  </a>
</div>

Step 2: Box detection and digit segmentation

<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images\result_image 2.png" alt="Logo" height="400" width="400">
  </a>
</div>
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images\result_image 3.png" alt="Logo" height="400" width="400">
  </a>
</div>
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images\result_image 4.png" alt="Logo" height="400" width="400">
  </a>
</div>
Step 3: Detection results form model
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images\result_image 5.png" alt="Logo" height="400" width="400">
  </a>
</div>
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images\result_image 6.png" alt="Logo" height="400" width="400">
  </a>
</div>
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images\result_image 7.png" alt="Logo" height="400" width="400">
  </a>
</div>

## References

Research paper:-

1. Hand gesture detection[https://www.researchgate.net/publication/284626785_Hand_Gesture_Recognition_A_Literature_Review](https://www.researchgate.net/publication/284626785_Hand_Gesture_Recognition_A_Literature_Review)
2. CNN pdf link: [https://arxiv.org/pdf/1909.08490]([https://arxiv.org/pdf/1909.08490]())
3. Box detection: [https://jwcn-eurasipjournals.springeropen.com/articles/10.1186/s13638-020-01826-x](https://jwcn-eurasipjournals.springeropen.com/articles/10.1186/s13638-020-01826-x)
4. Gaussian Blur and Otsu threshholding:[ https://www.researchgate.net/figure/Matched-image-after-applying-Gaussian-Blur-and-Otsus-thresholding-7-Set-of-ROI-To_fig2_323055012](https://www.researchgate.net/figure/Matched-image-after-applying-Gaussian-Blur-and-Otsus-thresholding-7-Set-of-ROI-To_fig2_323055012)
