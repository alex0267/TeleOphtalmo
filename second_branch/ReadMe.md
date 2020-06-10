# Glaucoma detection model - 2nd Branch

## General Description
This section has three parts.

1. We combine all the images from the Idrid dataset which are normally separated between a train and test folder. Three particular images of really damaged eyes are excluded from this union. After the combining all the images in one folder, we do an horizontal flip on each image and save the flip in the folder with the suffix "-flip" . All these operation are in the flip_image_idrid.ipynb notebook.

2. We train a Mask-Rcnn to recognize optic disks using our augmented Idrid dataset, and we then use the trained model to single out the optic discs in the Origa dataset.

3. We follow the same procedure as on the first branch to train a CNN to classify between healthy and glaucoma on our dataset of Origa optic discs.


## Data set used
Two datasets are used:

1. The Origa dataset which includes 168 fundus images of eyes presenting a glaucoma, and 482 fundus images of healthy eyes.

2. The Idrid dataset which includes 81 funds images and their respective masks. This dataset is doubled in size by applying a horizontal flip on the pictures and their respective masks

## Model & Training methods

1. Concerning the optic disk part, the model trained is Mask-RCNN pertained on COCO dataset.

2. Concerning the CNN part, the models trained are Resnet50 with pretrained weight obtained from the fastai library.

In the notebook three different training technique are used: freezing all layers, unfreezing all layers, and unfreezing only the last 2 layers. A confusion matrix is made at the end of each section to analyse the results.
