# Image-Classification---AWS-SageMaker.py
 
File Structure
data/: Directory for storing downloaded data.
train/, train_lst/, validation/, validation_lst/: Directories for preparing data for SageMaker.
image_classification.py: Main Python script for image classification on SageMaker.
README.md: Project documentation.
SageMaker Configuration
role: SageMaker execution role.
bucket_name: S3 bucket name.
training_image: SageMaker image for training.

Hyperparameters
Adjust hyperparameters in the script based on your image classification requirements.

Data Visualization
The script includes data visualization using Matplotlib to display random images from the dataset.

SageMaker Model Deployment
The trained model is deployed on SageMaker for making real-time predictions.
