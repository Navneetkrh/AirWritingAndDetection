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

-**Real-time Finger Detection**: Implement computer vision algorithms to detect and track finger movements in real-time using the webcam feed. The system should accurately recognize the position, orientation, and movement of fingers to enable precise interaction.

-**Virtual Canvas Creation**: Develop a virtual canvas where users can write and draw using their fingers.

-**Gesture Recognition**: Incorporate gesture recognition algorithms to interpret different finger gestures, such as using index finger to write on the virtuall canvas, or using the thumb to end the session.

-**Digit Recognition**: Train a machine learning model to recognize digits written by the user on the virtual canvas and convert it into digital numbers for further processing.

# Methodology

The methodology for the Air Writing and Detection project involves several key steps, leveraging computer vision techniques and machine learning models to enable users to write or draw on a digital canvas in real-time using their fingertips. The process can be broadly divided into the following sections:

### 1. Data Acquisition

The first step involves capturing video frames or images of handwritten digits drawn in the air using a camera or webcam. This data serves as the input for the subsequent stages of the system.

### 2. Preprocessing

Before feeding the captured data into the recognition and segmentation algorithms, it is essential to preprocess the data to improve its quality and remove any unwanted noise or background interference. Preprocessing techniques employed in this project include:

***Background Subtraction** : Separating the foreground (handwritten digits) from the background by using techniques such as frame differencing or background modeling.

***Noise Removal** : Applying filters or morphological operations to remove noise and improve the clarity of the handwritten digits.

***Contrast Enhancement** : Adjusting the contrast of the input image or video frames to enhance the visibility of the handwritten digits.

### 3. Air Writing Recognition

The air writing recognition component of the system is responsible for accurately classifying the handwritten digits drawn in the air. This stage involves the following steps:

1.**Feature Extraction** : Extracting relevant features from the preprocessed data that can effectively represent the handwritten digits. Common feature extraction techniques used in this project include:

* Histogram of Oriented Gradients (HOG)
* Scale-Invariant Feature Transform (SIFT)
* Local Binary Patterns (LBP)

1.**Model Training** : Training a machine learning model, such as a Convolutional Neural Network (CNN) or a Support Vector Machine (SVM), using labeled data for digit recognition. The choice of model and training parameters depends on the specific requirements and constraints of the project.

2.**Model Evaluation and Fine-tuning** : Evaluating the trained model's performance on test data and fine-tuning the hyperparameters or architecture if necessary to improve the recognition accuracy.

### 4. Image Segmentation

The image segmentation component of the system is responsible for separating individual digits from the input image or video frame. This is crucial for recognizing multiple digits in a single input and improving the overall accuracy of the system. The segmentation process involves the following steps:

1.**Connected Component Analysis** : Identifying connected regions or blobs in the preprocessed image that potentially represent individual digits.

2.**Contour Detection and Analysis** : Detecting and analyzing the contours or boundaries of the connected components to determine their shape and size characteristics.

3.**Digit Separation** : Separating and extracting individual digits from the input based on the contour analysis and connected component information.

4.**Postprocessing** : Performing additional operations, such as bounding box adjustment or region filtering, to refine the segmentation results.

### 5. Integration and User Interface

The air writing recognition and image segmentation components are integrated into a unified system with a user-friendly interface. The interface allows users to draw digits in the air using a camera or webcam and receive recognition results in real-time. The interface may also provide additional features, such as visualizing the segmented digits or displaying the recognition confidence scores.

### 6. Performance Evaluation and Optimization

The overall system performance is evaluated using appropriate metrics, such as accuracy, precision, recall, and processing time. Techniques like cross-validation or holdout testing are employed to ensure reliable performance evaluation.

If necessary, the system is optimized for real-time performance by employing techniques like parallelization, code optimization, or hardware acceleration (if applicable). This step ensures that the system can operate smoothly and responsively, making it suitable for practical applications.

# Implementation Details

This section provides detailed information about the implementation of the various components and functions used in the air writing and detection system.

### Data Acquisition

The data acquisition process involves capturing video frames or images of handwritten digits drawn in the air using a camera or webcam. The captured data is stored in a suitable format (e.g., video files or image sequences) for further processing.

#### Class: `VideoCapture`

This class is responsible for acquiring video frames from a camera or a pre-recorded video file. It provides methods to start and stop the video capture process, as well as retrieve individual frames.

*`__init__(self, source=0)`: Initializes the `VideoCapture` object with the specified video source (e.g., camera index or file path).

*`read(self)`: Reads the next frame from the video source and returns a boolean indicating success and the frame itself.

*`release(self)`: Releases the video capture resources and closes the video source.

#### Function: `capture_frames(source, num_frames, output_folder)`:

This function captures a specified number of frames from the given video source and saves them as individual image files in the output folder.

*`source`: The video source (e.g., camera index or file path).

*`num_frames`: The number of frames to capture.

*`output_folder`: The directory path where the captured frames will be saved.

### Preprocessing

The preprocessing stage involves several techniques to enhance the quality of the input data and prepare it for the subsequent recognition and segmentation stages.

#### Function: `background_subtraction(frame, background_model)`:

This function performs background subtraction on the input frame using a background model. It separates the foreground (handwritten digits) from the background.

*`frame`: The input frame or image.

*`background_model`: The background model used for background subtraction.

#### Function: `noise_removal(image, kernel_size)`:

This function applies noise removal techniques, such as filtering or morphological operations, to the input image.

*`image`: The input image.

*`kernel_size`: The size of the kernel used for filtering or morphological operations.

#### Function: `contrast_enhancement(image, alpha, beta)`:

This function enhances the contrast of the input image using linear contrast stretching.

*`image`: The input image.

*`alpha`: The scaling factor for contrast enhancement.

*`beta`: The constant offset for contrast enhancement.

### Air Writing Recognition

The air writing recognition component is responsible for accurately classifying the handwritten digits drawn in the air.

#### Class: `DigitRecognizer`

This class encapsulates the functionality of feature extraction, model training, and digit recognition.

*`__init__(self, model_path=None)`: Initializes the `DigitRecognizer` object with an optional pre-trained model path.

*`extract_features(self, image)`: Extracts relevant features from the input image using techniques like HOG, SIFT, or LBP.

*`train_model(self, training_data, labels)`: Trains a machine learning model (e.g., CNN or SVM) using the provided training data and labels.

*`predict(self, image)`: Predicts the digit present in the input image using the trained model.

#### Function: `evaluate_model(model, test_data, test_labels)`:

This function evaluates the performance of the trained model on the provided test data and labels, computing metrics such as accuracy, precision, and recall.

*`model`: The trained machine learning model.

*`test_data`: The test data samples.

*`test_labels`: The corresponding labels for the test data.

### Image Segmentation

The image segmentation component separates individual digits from the input image or video frame.

#### Class: `DigitSegmenter`

This class implements the image segmentation functionality, including connected component analysis, contour detection, and digit separation.

-`__init__(self, preprocess_func=None)`: Initializes the `DigitSegmenter` object with an optional preprocessing function.

-`preprocess(self, image)`: Preprocesses the input image using the specified preprocessing function, if provided.

-`connected_components(self, image)`: Performs connected component analysis on the input image to identify potential digit regions.

-`contour_analysis(self, image, components)`: Analyzes the contours of the connected components to extract shape and size characteristics.

-`segment_digits(self, image, contours)`: Separates and extracts individual digits from the input image based on the contour analysis.

-`postprocess(self, image, digits)`: Performs additional postprocessing operations, such as bounding box adjustment or region filtering, on the segmented digits.

#### Function: `draw_bounding_boxes(image, bboxes, color)`:

This function draws bounding boxes on the input image, highlighting the segmented digit regions.

-`image`: The input image.

-`bboxes`: A list of bounding box coordinates for the segmented digits.

-`color`: The color used to draw the bounding boxes.

### Integration and User Interface

The integration and user interface components combine the air writing recognition and image segmentation functionalities into a unified system with a user-friendly interface.

#### Class: `AirWritingApp`

This class represents the main application for air writing and detection, integrating the various components and providing a user interface.

-`__init__(self)`: Initializes the `AirWritingApp` object and sets up the necessary components (e.g., video capture, digit recognizer, digit segmenter).

-`run(self)`: Runs the main application loop, capturing frames, recognizing digits, and displaying the results.

-`update_display(self, frame, prediction, bboxes)`: Updates the user interface display with the input frame, predicted digits, and segmentation bounding boxes.

-`handle_user_input(self)`: Handles user input, such as keyboard or mouse events, for controlling the application.

#### Function: `create_gui(app)`:

This function creates and sets up the graphical user interface (GUI) for the air writing application.

-`app`: The `AirWritingApp` instance to be integrated with the GUI.

### Performance Evaluation and Optimization

The performance evaluation and optimization components ensure that the system operates efficiently and meets the real-time performance requirements.

#### Function: `cross_validate(model, data, labels, n_folds)`:

This function performs cross-validation on the provided model, data, and labels to evaluate its performance.

-`model`: The machine learning model to be evaluated.

-`data`: The input data samples.

-`labels`: The corresponding labels for the data samples.

-`n_folds`: The number of folds for cross-validation.

#### Function: `optimize_performance(system, target_fps)`:

This function optimizes the performance of the air writing and detection system to achieve the desired frame rate (frames per second).

-`system`: The air writing and detection system to be optimized.

-`target_fps`: The desired frame rate (frames per second) for real-time performance.

## Results and Discussion

In this section, provide a detailed analysis of the results obtained from the air writing and detection system. Include quantitative metrics, such as accuracy, precision, recall, and processing time, as well as qualitative observations and examples. Discuss the strengths and limitations of the developed system, and identify potential areas for improvement or future work.

## Conclusion

Summarize the main achievements and conclusions of the project. Highlight the key contributions and the significance of the developed air writing and detection system in the field of computer vision and its potential applications.

## References

List the relevant references, research papers, and resources used in the development of the air writing and detection system.

This report provides a comprehensive overview of the air writing and detection system, covering the problem statement, methodology, implementation details, results, and discussion. It serves as a detailed documentation of the project, allowing others to understand and potentially replicate or extend the work.
