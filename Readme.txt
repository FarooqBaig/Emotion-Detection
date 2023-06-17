# Emotion-Detection-using-CNN-Architecture
This repository contains code for emotion detection using Convolutional Neural Network (CNN) architecture. The implementation utilizes the Haar cascade classifier from the OpenCV library for face detection, and the CNN model is trained on two datasets: FER-2013 and CK+. The model achieves an accuracy of 89.91% on FER-2013 and 96.01% on CK+.

**Requirements**
To run the emotion detection code, you need to have the following dependencies installed:

Python (version 3.6 or higher)
OpenCV (version 4.0 or higher)
NumPy
TensorFlow (version 2.0 or higher)
Keras (version 2.0 or higher)

**Dataset**
The emotion detection model is trained on two datasets: FER-2013 and CK+.

FER-2013: The FER-2013 dataset contains 35,887 grayscale images of faces with seven different emotions: angry, disgust, fear, happy, sad, surprise, and neutral. Each image is 48x48 pixels.
CK+: The CK+ dataset consists of 593 labeled facial expressions images. It includes six basic emotions: anger, disgust, fear, happiness, sadness, and surprise.

**Haar Cascade Classifier**
The Haar cascade classifier is utilized for face detection in the input images. It is implemented using the OpenCV library. The pre-trained Haar cascade model detects faces and extracts the facial region, which is then fed into the CNN model for emotion detection.

**CNN Architecture**
The emotion detection model uses a CNN architecture to classify emotions from the facial images. The CNN consists of multiple convolutional layers, pooling layers, and fully connected layers. The architecture is trained and optimized using the FER-2013 and CK+ datasets.

**Usage**
Install the required dependencies mentioned above.

Clone this repository to your local machine.

Run the code, Save the model and finally Use the model for classification of emotions

The output will display the detected emotion on each face found in the input image or video.

**Results**
The emotion detection model achieved an accuracy of 89.91% on the FER-2013 dataset and 96.01% on the CK+ dataset. These accuracy values represent the performance of the trained model on the test sets of the respective datasets.



