# Load and pre-process AWS data

This repository loads eye pictures from an aws bucket to a sage maker notebook, and then applies cropping and Gaussian blur to the data.

The end goal is to develope a workflow for pre-processing eye pictures for an image classification model developped through transfer learning.

## I. Data Used

The data used for this excersise are in the S3 bucket "teleophtalmo-sagemaker". The original data can be dowloaded from https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid .

## II. Notebooks

The first notebook, **load-data.ipynb** loads the data from the S3 bucket "teleophtalmo-sagemaker" on to the instance tmp, unzip the data and save them on the instance disk.

The second notebook, **pre-processing-eyes.ipynb** process the image picture through cropping and Gaussian Blur. The functions used here are based on the following kernel by Kaggle user *Neuron Engineer* : https://www.kaggle.com/ratthachat/aptos-eye-preprocessing-in-diabetic-retinopathy 
