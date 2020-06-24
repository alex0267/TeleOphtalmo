# Glaucoma detection model - 1st Branch

## General Description
This section implement a CNN to classify fundus images of healthy eyes and ones affected by a glaucoma.

## Data set used
The Origa dataset is used, which includes 168 fundus images of eyes presenting a glaucoma, and 482 fundus images of healthy eyes.

## Model & Training methods
The models trained are Resnet50 with pretrained weight obtained from the fastai library.
In the notebook three different training techniques are used: 
- freezing all layers, 
- unfreezing all layers, 
- unfreezing only the last 2 layers. 
A confusion matrix is made at the end of each section to analyse the results.
