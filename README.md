# Load and pre-process AWS data

This repository includes code to :
- load eye pictures from an AWS bucket to an AWS SageMaker notebook.
- preprocess eye photos (crop + Gaussian blur).

The  goal is to develope a workflow for pre-processing eye pictures for an image classification model developped with transfer learning.

## I. Data Used

The data used in this example are in a S3 bucket called "teleophtalmo-sagemaker". 
The original data can be dowloaded from https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid

## II. Notebooks

The first notebook, **load-data.ipynb** loads the data from the S3 bucket "teleophtalmo-sagemaker" onto the instance tmp, unzip the data and save them on the instance disk.

The second notebook, **pre-processing-eyes.ipynb** pre-processes the images (cropping and Gaussian Blur). 
The functions used here are based on the following kernel by Kaggle user *Neuron Engineer* : https://www.kaggle.com/ratthachat/aptos-eye-preprocessing-in-diabetic-retinopathy 
