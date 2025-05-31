
### 🔗 Demo  
[![Streamlit App](https://img.shields.io/badge/Streamlit-Demo-red?logo=streamlit)](https://facial-landmark-app-28.streamlit.app/)

Click the link above to try out the Streamlit demo for this project.
# FACIAL-LANDMARK-DETECTION

This project focuses on detecting facial landmarks using deep learning techniques. The dataset used is the iBUG 300-W dataset, and the project leverages machine learning model to precisely identify key points on a face, such as eyes, nose, mouth, and chin.This repository includes all the code, prelearning steps(i.e small projects related to main project ) and documentation related to the project.
 
## OBJECTIVE
The goal of the project is to develop a highly efficient model capable of accurately detecting facial key points in input images. This model will support real-time inference for applications such as face recognition, emotion detection, virtual makeup, and augmented reality (AR).

## LEARNING PROCESS


1. The project began with an understanding of Deep Learning, Artificial Neural Networks, and Numpy, leading to the creation of a basic MNIST model from the scratch using only numpy.
2. Next, we studied various loss functions like cross-entropy loss and optimizers like Adam, then learned PyTorch and built an MNIST model using this framework.
3. This was followed by exploring Convolutional Neural Networks (CNNs) and studying various CNN architectures, culminating in the implementation of a CIFAR-10 model.
4. We then created a custom dataset for facial landmark detection.
5. Finally, we learned and implemented the facial landmark detection algorithm using the PyTorch framework.


### MNIST-HANDWRITTEN DIGITS 
 - The MNIST AND MNIST-HANDWRITTEN DIGITS datasets are used. These datasets contain 60,000 training samples and 10,000 test samples. Each sample in the MNIST dataset is a 28x28 pixel grayscale image of a single 
   handwritten digit between 0 & 9.
 - Implemented using numpy and pytorch with ANN also but here mentioned is with pytorch and CNN.

![image](https://hackmd.io/_uploads/ByroZQK11x.png)



#### HYPERPARAMETERS

#### This is for the MNIST using the CNN
| Hyperparameters | Value | 
| -------- | -------- | 
|   Batch-Size| 500
Learning-rate	|0.001	
num of Epochs|50
Loss |      Cross Entropy Loss
Optimizer |    Adam Optimizer


### RESULTS
* Accuracy
  - 99.8% on the Train set
  - 99% on the Test set

* Visualization
![image](https://hackmd.io/_uploads/SJPVGXFkJx.png)
* Graphs
![image](https://hackmd.io/_uploads/HJ9zLQYykx.png)


### CIFAR-10
The CIFAR-10 dataset is used. This dataset contains 60,000 samples, with 50,000 training samples and 10,000 test samples. Each sample in the CIFAR-10 dataset is a 32x32 pixel color image belonging to one of 10 different classes, such as airplanes, cars, birds, cats, and more.

![image](https://hackmd.io/_uploads/H1Ub4XKkkx.png)

#### HYPERPARAMETERS
| Hyperparameters | Value | 
| -------- | -------- | 
|   Batch-Size| 1024
Learning-rate	|0.0001
num of Epochs|100
Loss |      Cross Entropy Loss
Optimizer |    Adam Optimizer
#### RESULTS
* Accuracy
  - 89.66 % on the Trainset
  - 76.47 % on the Testset
  - Accuracy is less in this because CIFAR-10 is large dataset which required the deep neutral network for the training but we implemented it using simple CNN.




* Graphs
  - ![image](https://hackmd.io/_uploads/ryNRL7YJkl.png)
  - ![image](https://hackmd.io/_uploads/HyZQD7Ykyg.png)
### CUSTOM-DATASET
In this prelearning task I learned about the custom dataset . How to create the our dataset from the given images . In deep learning it is most important step as we need to load first the data and then need to feed it to the model . So from this i get the idea how to load the data and make the dataset which can be feed into the model . This task is done on the BSD-100 dataset .

#### RESULTS
* Output
 - ![image](https://hackmd.io/_uploads/S1GGjmYkkg.png)
## MAIN PROJECT
### FACIAL LANDMARK DETECTION

For facial landmark detection, datasets such as the iBUG 300-W and similar facial landmark datasets are commonly used. These datasets contain a variety of images annotated with facial key points. For example, the iBUG 300-W dataset consists of thousands of images, with each image labeled with 68 facial landmarks, including points around the eyes, nose, mouth, and jawline. The dataset is typically divided into training and test sets, enabling model training and evaluation for tasks like face recognition, emotion analysis, and augmented reality applications.

#### PROCEDURE




##### **DATA PROCESSING AND AUGMENTATION**

For this facial landmark detection project, a comprehensive data preprocessing and augmentation pipeline is used to enhance the quality and diversity of the training data. The steps involved in the preprocessing pipeline are described below:

**1. FaceLandmarksAugmentation Class**
This class is designed to apply several transformations to the input images and their corresponding facial landmarks to improve the model’s robustness. The transformations applied include:

 - Color Jittering: Adjusts the brightness, contrast, saturation, and hue of the image to account for different lighting conditions. This is done using 
  transforms.ColorJitter.

 - Offset Cropping: The image is cropped based on given facial crop coordinates, and padding is added around the face. The landmarks are also adjusted to reflect 
  the cropped image's new coordinate system.

 - Random Face Cropping: The image is randomly cropped, and the landmarks are adjusted to fit the new image dimensions. This helps the model generalize to various 
  face positions and scales.

 - Random Rotation: A random rotation angle is applied to the image and the landmarks. The landmarks are transformed using a rotation matrix to account for the 
   angle applied to the image. This ensures that the model can handle different head poses.

**2. Preprocessor Class**
The Preprocessor class combines all the augmentation steps for easy use during data loading. The preprocessing steps include:

 - Offset Cropping: Applied to ensure that the face is correctly centered and cropped.

 - Random Face Cropping: Further crops the image randomly to add variability to the training data.

 - Random Rotation: Rotates the image and corresponding landmarks by a random angle.

 - Grayscale Conversion: Converts the image to grayscale, as the training set consists of grayscale images.

 - Normalization: The image is normalized to the range [-1, 1] to improve convergence during model training. Landmarks are also normalized to be relative to the 
  image dimensions and shifted to the range [-0.5, 0.5].

**3. datasetlandmark Class**
The dataset class is responsible for loading the images, their corresponding landmarks, and facial crop coordinates from the iBUG 300-W dataset. The key steps include:

 - XML Parsing: The dataset annotations are stored in XML format. The XML tree is parsed to extract the image paths, facial landmark coordinates, and facial crop 
  coordinates.

 - Data Loading: Each image is loaded along with its landmarks. The preprocessing pipeline (Preprocessor) is applied to the image and landmarks before returning 
  them as a tensor ready for the model.

**Workflow:**
 - Image Augmentation: Each image undergoes color jittering, cropping, and rotation to make the model robust to lighting variations, head poses, and different 
   face positions.
 - Landmark Adjustment: As the image is transformed, the corresponding facial landmarks are also adjusted to ensure that they are correctly aligned with the 
   transformed image.
 - Normalization: Both the image and landmarks are normalized to improve model performance and training stability.
   This process ensures that the model is trained on a wide variety of images with different lighting conditions, face positions, and orientations, making it more 
   robust and capable of accurately detecting facial landmarks in real-world scenarios.And ready tofeed it into the model for the training.

![image](https://hackmd.io/_uploads/HkrIMVtk1l.png)
![image](https://hackmd.io/_uploads/HJco7EYy1e.png)


#### **ARCHITECTURE(NETWORK)**

In traditional convolution layers, a single convolution kernel is tasked with simultaneously mapping cross-channel correlations and spatial correlations.
Inception modules makes this process more efficient by implementing parallel convolutions with varying filter sizes (e.g., 1x1, 3x3, 5x5), allowing the network to capture a variety of features.
Xception takes this idea further by pushing the separation of cross-channel and spatial correlations to the extreme.
- **Hypothesis behind Xception architecture:** 
     The mapping of cross-channels correlations and spatial correlations in the feature maps of CNNs can be entirely decoupled.
  	This decoupling is achieved in the following way:
  - 1.Depthwise convolution: Operates on each channel independently rather than dividing them into segments, extracting spatial correlations.  
  - 2.Pointwise convolution (1x1): Combines the outputs from the depthwise step, learning cross-channel correlations.

![Screenshot (69)](https://github.com/user-attachments/assets/cfc46024-6a83-453e-80b9-840de63fd61b)
![image](https://github.com/user-attachments/assets/b96be151-b01f-443d-825f-0365aba97f62)


#### **TRAINING LOOP**
The training loop in this project is designed to ensure efficient learning and evaluation of the facial landmark detection model. Below are the key components and steps involved:

 - 1. Epochs and Batches
The model is trained for 30 epochs, where each epoch represents a complete pass through the training data. The train_data is divided into batches, and for each batch, the model processes a subset of the data to update its weights.

 - 2. Loss Calculation
For each batch, the model computes the output (predicted landmarks) based on the input image features. The loss is calculated using an objective function that measures the difference between the predicted and actual facial landmarks. In each epoch:

   - Backward Propagation: The loss is used to compute gradients, which are then used by the optimizer to update the model’s weights.
   - Gradient Zeroing: After each update, the gradients are reset to prevent accumulation over batches.
 - 3. Validation
After each epoch, the model’s performance is evaluated on a separate validation dataset using the validate() function. The validation loss is calculated, providing a metric to monitor overfitting and generalization to unseen data.

 - 4. Checkpointing
To safeguard training progress and ensure the best model is saved, a checkpointing system is implemented:

   - Saving Every Epoch: At the end of each epoch, the model’s state (including its parameters, optimizer state, and the current epoch) is saved to a checkpoint file. This allows for recovery in case of an interruption.

   - Best Model Saving: If the validation loss at the current epoch is lower than the previous best, the model is saved as the best model. This ensures that the model with the best performance on validation data is preserved, even if the training continues.

 - 5. Monitoring Training
During training, the loss values for both the training and validation datasets are logged and printed at the end of each epoch. This provides real-time feedback on how the model is learning:

    - Training Loss: The average loss across all batches in the training set.
    - Validation Loss: The loss calculated on the validation set, used to monitor generalization.

#### HYPERPARAMETERS
| Hyperparameters | Value | 
| -------- | -------- | 
|   Batch-Size| 32
Learning-rate	|0.00075
num of Epochs|30
Loss |      MSE LOSS
Optimizer |    Adam Optimizer
#### RESULTS
- ACCURACY : Depends on the threshold we can not get directly ,first we have define threshold and need to compute the euclidean distance between the predicted and true landmarks and if it is less than threshold then we can say it is accurate.
- HOW THE MODEL LEARNING:
 
![image](https://github.com/user-attachments/assets/fd37a0f7-00bf-47aa-9050-64ec451aacbf)
![image](https://github.com/user-attachments/assets/c3939287-80ff-4587-8253-e6536923463f)




- GRAPHS :
![image](https://hackmd.io/_uploads/S1qu_VYkyg.png)


### SOFTWARE TOOLS USED 
- Python
- PyTorch
- CUDA
- Numpy
- PyTorch
- PIL
- Matplotlib


  
