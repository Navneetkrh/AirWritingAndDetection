# CSL7360: Computer Vision

# Project: Air Writing and Detection

# Team Members

| Name          | Roll No  |

|-------------|--------|

| Navneet Kumar | B21CS050 |

| Ayush Jain    | B21BB006 |

| Soham Parikh  | B21CS074 |

| Het Dave      | B21CS020 |

## How To Run The Code

There are two ipynb files in the repository. One is for training and saving of our digit Recogniton Model and the other is for the Air Writing and Detection, which utilizes this trained model.

The trained model for digit recognition is saved in the file `model.h5`. This file is used in the `Air_writing+recognition.ipynb` file.

You need to install libraries, which can be done by running the following command in the terminal:

``pip install -r requirements.txt``at

To run the code you can click run all cells in the `Air_writing+recognition.ipynb` file. This will open the webcam and you can start writing using your index finger. The code will detect the digits you write in the air and display them on the screen with the predicted output form the model.

To retrain or make changes to the model or dataset, you can run the `Digit_Recognition.ipynb` file. This will train the model and save it in the file `model.h5`. This file can then be used in the `Air_writing+recognition.ipynb` file.

## Problem Statement

After coronavirus outbreak,we had a couple of changes in our daily lifstlye to get used with.

We got to know the importance of online plateforms however for colaboration and learning purposes we need to have a whiteboard to write and explain things to others.

In today's digital age, with the increasing demand for remote collaboration and interactive learning, there is a growing need for innovative solutions that enable users to annotate and interact with virtual content seamlessly. One such application is a Virtual Whiteboard system that leverages computer vision techniques to detect fingers and allows users to write or draw on a digital canvas in real-time using their fingertips.

The goal of this project is to develop a robust Virtual Whiteboard system that utilizes a webcam to detect finger movements accurately and translate them into digital ink on a virtual canvas. This system should provide an intuitive and interactive interface for users to write, draw, and interact with the content without the need for physical input devices such as a stylus or mouse.

Key Objectives:

-**Real-time Finger Detection**: Implement computer vision algorithms to detect and track finger movements in real-time using the webcam feed. The system should accurately recognize the position, orientation, and movement of fingers to enable precise interaction.

-**Virtual Canvas Creation**: Develop a virtual canvas where users can write and draw using their fingers.

-**Gesture Recognition**: Incorporate gesture recognition algorithms to interpret different finger gestures, such as using index finger to write on the virtuall canvas, or using the thumb to end the session.

-**Digit Recognition**: Train a machine learning model to recognize digits written by the user on the virtual canvas and convert it into digital numbers for further processing.

# Methodology

The methodology for the Air Writing and Detection project involves several key steps, leveraging computer vision techniques and machine learning models to enable users to write or draw on a digital canvas in real-time using their fingertips. The process can be broadly divided into the following sections:

### 1. Data Acquisition

The first step involves capturing video frames or images of handwritten digits drawn in the air using a camera or webcam. This data serves as the input for the subsequent stages of the system. We gather data form the user using class named as 'writer' which tracks the points on the canvas on which user draws and store the information for each object created after shift from 'non_pointing state' to 'pointing state'

### 2. Preprocessing

Before feeding the captured data into the model, it is essential to preprocess the data to improve its quality and remove any unwanted noise or background interference. Preprocessing techniques employed in this project include:

***Background Subtraction** : Separating the foreground (handwritten digits) from the background for better results expectation from the model .

***Noise Removal** : Applying Gaussian filters to remove noise and improve the clarity of the handwritten digits.

***Image cropping** : Picking up desired digit from the image using image slicing method 

### 3. Air Writing Recognition

The air writing recognition component of the system is responsible for accurately classifying the handwritten digits drawn in the air. This stage involves the following steps:

1.**Feature Extraction** : Extracting relevant features from the preprocessed data that can effectively represent the handwritten digits. Common feature extraction techniques used in this project include:

* Guassian_filter for removing the noise form the image 
* Digit Block Segmentation on the Canvas by thresholding mechnanism
* Point tracking mechnaism

1.**Model Training** : Training a machine learning model on a Convolutional Neural Network (CNN) using labeled data for digit recognition on MNIST dataset.The model accuracy of the network on testing datset is more than 97%.

2.**Model Evaluation and Fine-tuning** : Evaluating the trained model's performance on test data and fine-tuning the hyperparameters or architecture and finally training on best result parameters.

### 4. Image Segmentation (This is one of the key feature of the project)

The image segmentation component of the system is responsible for separating individual digits from the input image frame captured. This is crucial for recognizing multiple digits in a single input and improving the overall accuracy of the system. It helps the user to re-edit the digit or complete the digit if left incompleted in one go. The segmentation process involves the following steps:

1.**Connected Component Analysis** : Identifying connected regions or blobs in the preprocessed image that potentially represent individual digits.

2.**Contour Detection and Analysis** : Detecting and analyzing the contours or boundaries of the connected components to determine their shape and size characteristics.

3.**Digit Separation** : Separating and extracting individual digits from the input based on the contour analysis and connected component information.

4.**Postprocessing** : Performing additional operations such as bounding box adjustment through threshhold management and region filtering to refine the segmentation results.

### 5. Integration and User Interface

The air writing recognition and image segmentation components are integrated into a unified system with a user-friendly interface. The interface allows users to draw digits in the air using a camera or webcam and receive recognition results as code output. The interface may also provide additional features such as visualizing the segmented digits or displaying the recognition probablity scores for digits from 0-9.

### 6. Performance Evaluation and Optimization

The overall system performance is evaluated by the user by final visualization and predction from the algrithm. We have used 2 methods for digit extraction form canvas: 
1) Image cropping and padding 
2) Image egmentation though contours detection

### 7. Additional functionality:-
We have added algorithm to additionally detect enclosed area drawn on the canvas. This algorithm first applies Gaussian blur to the image, and then employs a Canny edge detector along with detection of contours, to finally provide outlines of the enclosed areas with image canvas as boundary.

# Implementation Details

This section provides detailed information about the implementation of the various components and functions used in the air writing and detection system.

### Data Acquisition

The data acquisition process involves capturing video frames or images of handwritten digits drawn in the air using a camera or webcam. The captured data is stored in a suitable format (e.g., video files or image sequences) for further processing.

#### Class: `VideoCapture`

This class is responsible for acquiring video frames from a camera or a pre-recorded video file. It provides methods to start and stop the video capture process, as well as retrieve individual frames.

*`__init__(self, source=0)`: Initializes the `VideoCapture` object with the specified video source (e.g., camera index or file path).

*`read(self)`: Reads the next frame from the video source and returns a boolean indicating success and the frame itself.

*`release(self)`: Releases the video capture resources and closes the video source.

### Preprocessing

The preprocessing stage involves several techniques to enhance the quality of the input data and prepare it for the subsequent recognition and segmentation stages.


### Air Writing Recognition

The air writing recognition component is responsible for accurately classifying the handwritten digits drawn in the air.

#### Class: `DigitRecognizer`

This class encapsulates the functionality of feature extraction, model training, and digit recognition.

*`__init__(self, model_path=None)`: Initializes the `DigitRecognizer` object with an optional pre-trained model path.

*`extract_features(self, image)`: Extracts relevant features from the input image using techniques like HOG, SIFT, or LBP.

*`train_model(self, training_data, labels)`: Trains a machine learning model (CNN) using the provided training data and labels.

*`predict(self, image)`: Predicts the digit present in the input image using the trained model.

#### Function: `evaluate_model(model, test_data, test_labels)`:

This function evaluates the performance of the trained model on the provided test data and labels, computing metrics such as accuracy, precision, and recall.

*`model`: The trained machine learning model.

*`test_data`: The test data samples.

*`test_labels`: The corresponding labels for the test data.

### Image Segmentation

The image segmentation component separates individual digits from the input image or video frame.

### Integration and User Interface

The integration and user interface components combine the air writing recognition and image segmentation functionalities into a unified system with a user-friendly interface.

## Results and Discussion

In this section, provide a detailed analysis of the results obtained from the air writing and detection system. Include quantitative metrics, such as accuracy, precision, recall, and processing time, as well as qualitative observations and examples. Discuss the strengths and limitations of the developed system, and identify potential areas for improvement or future work.

## Conclusion

Summarize the main achievements and conclusions of the project. Highlight the key contributions and the significance of the developed air writing and detection system in the field of computer vision and its potential applications.

## References

List the relevant references, research papers, and resources used in the development of the air writing and detection system.

This report provides a comprehensive overview of the air writing and detection system, covering the problem statement, methodology, implementation details, results, and discussion. It serves as a detailed documentation of the project, allowing others to understand and potentially replicate or extend the work.
