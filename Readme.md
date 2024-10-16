# Autism Detection Project using QCNN

[Dataset Link](https://drive.google.com/drive/folders/1kFSl8acOOQLJwG3v9Sdx2Q-TKpkA_U8b) 

## Classical Neural Network Approach:

This project aims to detect autism traits in individuals by analyzing facial features using a Convolutional Neural Network (CNN). The model is trained on a dataset of facial images with various augmentations to improve robustness and accuracy. The goal is to achieve a high detection accuracy of approximately 90%.

The project implements a CNN-based classifier that uses a dataset of preprocessed images. It handles class imbalance using SMOTE (Synthetic Minority Over-sampling Technique) and applies data augmentation to prevent overfitting. The final model achieves enhanced performance with careful hyperparameter tuning and regularization techniques.




### **Features:**

**Convolutional Neural Network (CNN):** 
A CNN architecture designed for image classification, optimized for detecting autism traits.

**Data Augmentation:** 
Uses techniques like rotation, zooming, flipping, and brightness adjustment to make the model more robust to variations in images.

**Class Imbalance Handling:**
SMOTE is applied to resample the training dataset and mitigate the effects of class imbalance.

**Early Stopping & Checkpointing:** 
Prevents overfitting by stopping training when validation performance stops improving and saves the best model.

**Hyperparameter Optimization:** 
Hyperparameters like learning rate, batch size, and epochs are tuned to achieve maximum accuracy.

### How It Works

1. **Data Preprocessing:**
    Normalization: The pixel values of images are normalized to a range of 0 to 1.
    Data Augmentation: The training data is augmented using Keras' ImageDataGenerator, applying transformations such as rotation, flipping, zooming, and shifting.
    SMOTE Resampling: The class imbalance is handled using SMOTE, ensuring the minority class is not underrepresented.

2. **CNN Model:**

    Architecture: The CNN consists of multiple convolutional layers with ReLU activation followed by max-pooling layers. The final layers include a fully connected network with a softmax output for binary classification (Autism/No Autism).
    Regularization: Dropout is applied to prevent overfitting during training.
    Optimizer: Adam optimizer is used with a learning rate of 0.001 for efficient training.

3. **Training:**

    The model is trained using the ImageDataGenerator on the resampled and augmented dataset.
    Early stopping ensures that training halts when validation accuracy stops improving, while a model checkpoint saves the best version of the model during training.



## Quantum Convolutional Neural Network (QCNN) for Autism Classification:

This project showcases the implementation of a Quantum Convolutional Neural Network (QCNN) using PennyLane and TensorFlow. The aim is to classify images into two categories: "Autistic" and "Non_Autistic," leveraging quantum-enhanced layers to explore the potential of hybrid quantum-classical machine learning.

### Key Features:

**1. Hybrid Model:** Combines classical deep learning with quantum layers to potentially improve the model's performance in image classification tasks.

**2. Custom Dataset:** Uses an autism dataset with pre-labeled images for binary classification.

**3. Performance Comparison:** Provides a side-by-side comparison of model accuracy and loss between quantum and non-quantum models.

**4. Visualization:** Includes detailed plots to visualize model performance across training epochs.


### Dependencies:
- PennyLane
- TensorFlow
- Keras
- Pillow
- Matplotlib

### How It Works:

- The project initializes a simple CNN using Keras. Two versions of the model are trained: one with a quantum layer and one without.
- The dataset is preprocessed, and the images are converted into a format suitable for training.
- The models are trained for 30 epochs, with the results saved and visualized.

### Results:
After training, the notebook outputs a comparison of accuracy and loss for both the quantum-enhanced and classical models, helping 
assess the impact of quantum layers on performance.

**Installation:**

**Prerequisites:**
To run this project, you'll need to install the following dependencies:

- Python 3.11+
- TensorFlow 2.17
- Keras
- Scikit-learn
- Imbalanced-learn

You can install the required packages using the provided requirements.txt file:
`pip install -r requirements.txt`

### Contributing

Contributions to the project are welcome! To contribute:

1. Fork the repository.
2. Create a new feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some amazing feature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.


[![Watch the video](https://i.sstatic.net/iI3WN.png)](https://youtu.be/Q0GBxvunFWA)


### License
This project is licensed under the MIT License - see the LICENSE file for details.








