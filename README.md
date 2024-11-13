# Garbage Classification Model - Programming Assignment

Names: 
Felipe Castano - 30259693

Wooju Chung - 30263578

This project implements a garbage classification model using PyTorch to categorize waste into "Black," "Blue," "Green," and "TTR" categories. Classification is based on images and text descriptions, helping to identify the correct category for each waste item. This system was developed as part of **ENEL 645** at the University of Calgary as an exercise to assist with waste management in the city.

## Project Description

This project aims to solve the problem of garbage classification by combining image processing and natural language understanding to interpret user-provided waste descriptions. We use a Convolutional Neural Network (ResNet50) for image analysis and BERT for text analysis. Outputs from both models are combined and passed through fully connected layers for final classification. This solution helps residents correctly categorize their waste, contributing to environmental sustainability.

We first tried to solve the problem by using k-fold stratified, but after several models and attempts, we gave up on this model. At this point we decided to integrate the Resnet50 and Bert in a fully connected layer and concatenate the characteristics of both to obtain the expected classification (following the recommendations given by the professor from the question I asked during the class).

### Key Requirements

To run it, you'll need Python 3.8+ and essential libraries, including PyTorch, Transformers, and Torchvision for model development; Pillow and Matplotlib for image handling and visualization; and Scikit-Learn for evaluation metrics. CUDA-enabled GPUs are recommended for faster training, though they are optional. The data must be organized in labeled folders for efficient loading and processing by the model's custom dataset class.

- `torch`: Deep learning framework for building and training the model.
- `torchvision`: Contains pretrained models and image transformation tools.
- `transformers`: NLP models library, used here to load BERT.
- `sklearn`: Metric and evaluation tools.
- `Pillow`: Image manipulation.
- `matplotlib`: Data visualization and graphing.
- `Scikit-Learn`: Evaluation metrics.
- `Seaborn`: Data visualization (confusion matrix).

## Usage

### Training and Testing the Combined Model

This code integrates both image and text data to classify garbage items into the categories "Black," "Blue," "Green," and "TTR". Using a combined model, it processes images with a ResNet50 architecture and text descriptions with a BERT model, training both together to optimize classification accuracy.

The code structure includes:
The code structure is organized as follows:

- **Imports and Configuration**: Essential libraries are imported, device (CPU/GPU) is set, and data paths are specified.
- **Dataset Definition**: A custom GarbageDataset class is created to load images and text, applying image transformations and text tokenization for multi-modal input.
- **Model Definition**: The MultiModalModel class is defined, combining a ResNet50 for image features and a BERT model for text features, followed by fully connected layers for final classification.
- **Data Transformations and Tokenizer**: Image transformations for data augmentation and the BERT tokenizer are initialized.
- **Data Loading**: Training, validation, and test datasets are loaded using DataLoader, with batch sizes and shuffle settings.
- **Training Components**: Loss function, optimizer, and learning rate scheduler are set up. Class weights are computed to address class imbalance.
- **Training Function**: A train_model function trains the model, tracks validation accuracy, and implements early stopping for optimization.
- **Evaluation Function**: evaluate_model calculates accuracy, and confusion matrix, and displays incorrect classifications, saving visual results to files.
- **ROC Curve Plotting**: plot_roc_curve generates ROC curves for multi-class evaluation, saved as an image.

### Data Access

The data is accessed through specified directory paths for training, validation, and testing (TRAIN_PATH, VAL_PATH, and TEST_PATH). Within these directories, images and text files are organized in labeled folders corresponding to each classification category (e.g., "Black," "Blue," "Green," "TTR"). The custom GarbageDataset class reads each image and associated text description, transforming images and tokenizing text for model input. For each label folder, image files (.jpg or .png) and their corresponding text files (.txt) are processed, enabling the model to access multi-modal inputs.

## Result

The model successfully classified garbage images and text descriptions, achieving a final evaluation accuracy of approximately 72%. This result reflects the effectiveness of its multi-modal approach, combining ResNet50 for image feature extraction and BERT for text processing, enabling the model to learn patterns from both data types. Through early stopping, the model retained the best-performing version based on validation accuracy, mitigating potential overfitting. Overall, the model provides a reliable, scalable framework for image-text classification with room for optimization.
