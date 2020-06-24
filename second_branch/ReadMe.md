# Glaucoma detection model - 2nd Branch

## General Description
The 2nd branch aims at capturing glaucoma specific features with a Resnet50 model, by cropping images, and by forcing the model to focus on the region of interest : the optic disk. In order to crop the images automatically, we train a Mask-RCNN to recognize the optic disks.
The 2nd branch process has three steps:
1. Pre-processing for the Mask R-CNN : the IDRID dataset initially comes with 2 folders : train and test datasets. We combine all the images into one folder. 3 images of really damaged eyes are excluded from this union. After combining all the images, we do an horizontal flip on each image and save the flip in another folder with the suffix "-flip". This pre-processing step in the notebook flip_image_idrid.ipynb.
2. Optic disk selection : we train a Mask-RCNN to recognize optic disks using our augmented Idrid dataset, and we then use the trained model to isolate the optic disks in the ORIGA dataset (SANAS + GLAUCOMA). The optic disk selection training step is in the notebook train_Mask_RCNN_optic_disc.ipynb. The optic selection step is in the notebook ???.
3. Classification : we use the same methodology as the first branch to train a Resnet50 model to classify between healthy and glaucoma on the cropped ORIGA dataset.
The classification step in the notebook ???.

## Data set used
Two datasets are used:
1. Optic disk selection : the IDRID dataset which includes 81 funds images and their respective masks. This dataset is doubled in size by applying a horizontal flip on the pictures and their respective masks.
2. Classification : The ORIGA dataset which includes 168 fundus images of eyes presenting a glaucoma, and 482 fundus images of healthy eyes.

## Model & Training methods
1. Optic disk selection : pretained Mask-RCNN model on COCO dataset. Link here : 
2. Classification : Resnet50 with pretrained weight obtained from the Fast.ai library. In the notebook 3 different training techniques are used: 
- freezing all layers, 
- unfreezing all layers, 
- unfreezing only the last 2 layers. 
A confusion matrix is made at the end of each section to analyse the results.
