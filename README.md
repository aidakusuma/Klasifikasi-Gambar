# Dog and Cat Image Classification Project

## Project Overview
This is a deep learning project that uses a Convolutional Neural Network (CNN) to classify images of dogs and cats with high accuracy. The project leverages transfer learning with the VGG16 pre-trained model and implements several advanced techniques to improve model performance.

## Dataset
- Source: Kaggle Dog and Cat Classification Dataset Downloading from https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset

Path to dataset files: /root/.cache/kagglehub/datasets/bhavikjikadara/dog-and-cat-classification-dataset/versions/1
- Total Images: 24,998
- Categories: Dogs and Cats
- Image Quality Processing:
  - Removed corrupted images
  - Filtered out images smaller than 32x32 pixels
  - Final clean dataset: 19,995 training images, 4,998 validation images

## Model Architecture
- Base Model: VGG16 (pre-trained on ImageNet)
- Transfer Learning Approach:
  - Frozen base model weights
  - Added custom layers:
    - Convolutional layers
    - Max pooling
    - Dropout for regularization
    - Dense layers with ReLU activation
- Final Layer: Sigmoid activation for binary classification

## Data Augmentation
Applied ImageDataGenerator with the following transformations:
- Rescaling pixel values
- Random rotations
- Width and height shifts
- Shear and zoom transformations
- Horizontal flipping

## Training Parameters
- Optimizer: Adam
- Learning Rate: 0.0001 (with dynamic reduction)
- Loss Function: Binary Crossentropy
- Batch Size: 32
- Image Size: 150x150 pixels

## Key Callbacks
1. Early Stopping
   - Monitors validation accuracy
   - Stops training if no improvement
2. Learning Rate Reduction
   - Dynamically reduces learning rate
3. Custom Accuracy Callback
   - Stops training if 96% validation accuracy achieved

## Performance Metrics
- Peak Training Accuracy: ~94.91%
- Peak Validation Accuracy: ~91.32%
- Minimal overfitting
- Consistent performance across epochs

## Model Conversion and Deployment
Multiple model formats generated:
- Keras (.h5)
- TensorFlow.js 
- SavedModel
- TensorFlow Lite

## Requirements
- Python 3.10
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pillow
