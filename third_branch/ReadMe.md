# Glaucoma detection model - 3rd Branch

## General Description
This section has two parts.

1. Using the notebook image_segmentation.ipynb we create masks for the optic discs and the cup of the images in the Magrabia dataset.

2. We train a FCN model on our created dataset through the notebook fcn_for_ratio.ipynb.


## Data set used
One datasets is used:

1. Magrabia, which consist of 47 male fundus pictures and 48 female d'indus pictures each with 6 pictures where a different ophtalmologiste draw the optic disc and the cup of the eye.

## Model & Training methods

The model used is a FCN (fcn_resnet101 from pytorch).
