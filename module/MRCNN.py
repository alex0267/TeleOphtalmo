import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Tuple

import mrcnn.model as modellib
import numpy as np
import pandas as pd
import tensorflow as tf
from helpers import (COLORS, DetectorDataset, create_cropped_image,
                     create_pathology_dataframe, mmod, mrcnn_iou_eval)
from keras.backend.tensorflow_backend import set_session
from mrcnn.config import Config as MRCNNConfig
from skimage.io import imread
from sklearn.model_selection import train_test_split


@dataclass
class CroppedImageConfig:
    """Configuration for the image cropper.

    :param INPUT_PATH_GLAUCOMA: path to the folder containing the glaucoma
        images
    :param INPUT_PATH_HEALTHY: path to the folder containing the healthy
        images
    :param NAME_GLAUCOMA: base name for the cropped glaucoma images
    :param NAME_HEALTHY: base name for the cropped healthy images
    :param OUTPUT_PATH_GLAUCOMA: path to a folder to store the cropped
        glaucoma images.
    :param OUTPUT_PATH_HEALTHY: path to a folder to store the cropped
        healthy images.
    """

    INPUT_PATH_GLAUCOMA: str = ""
    INPUT_PATH_HEALTHY: str = ""
    NAME_GLAUCOMA: str = ""
    NAME_HEALTHY: str = ""
    OUTPUT_PATH_GLAUCOMA: str = ""
    OUTPUT_PATH_HEALTHY: str = ""


@dataclass
class Config:
    """MRCNN configuration to pass to the MRCNN model constructor.

    :param IS_INFERENCE: is the model being used in inference mode
    :param USE_GPU: should the GPU be used
    :param DEBUG: if True, shorten train time (epochs etc)
    :param WIDTH: input image width
    :param NUM_CLASSES: number of classes to detect
    :param MASK_PATHS: dictionary with keys being mask class names and
        values being the path to the folder containing the mask files
    :param MASK_COLOR: color of the masks in the mask files
    :param IMAGE_PATH: path to the folder containing the input images
        dataset.
    :param WEIGHTS_PATH: path to the weights to load in inference mode
    :param MODEL_DIR: path to the directory to save the models to
        in train mode
    :param LEARNING_RATE: learning rate
    :param EPOCHS: number of epochs to train for
    :param cropped_image: configuration for the images to crop"""

    IS_INFERENCE: bool = False
    USE_GPU: bool = True
    DEBUG: bool = True
    WIDTH: int = 1024
    NUM_CLASSES: int = 2
    MASK_PATHS: Dict[str, str] = field(default_factory=dict)
    MASK_COLOR: int = COLORS["red"]
    IMAGE_PATH: str = ""
    WEIGHTS_PATH: str = ""
    MODEL_DIR: str = ""
    LEARNING_RATE: float = 0.0001
    EPOCHS: int = 50
    cropped_image: CroppedImageConfig = CroppedImageConfig()


class MyMRCNNConfig(MRCNNConfig):
    """Private MRCNN configuration. This class is not to be instanciated outside
    of the MRCNN class."""

    # Give the configuration a recognizable name
    NAME = "Idrid"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    BACKBONE = "resnet50"

    RPN_ANCHOR_SCALES = (16, 64, 128, 256, 512)
    TRAIN_ROIS_PER_IMAGE = 64
    #     MAX_GT_INSTANCES = 14
    #     DETECTION_MAX_INSTANCES = 10
    DETECTION_MIN_CONFIDENCE = 0.75
    #     IMAGE_RESIZE_MODE = "crop"
    DETECTION_NMS_THRESHOLD = 0.0

    # balance out losses
    LOSS_WEIGHTS = {
        "rpn_class_loss": 0.1,
        "rpn_bbox_loss": 0.1,
        "mrcnn_class_loss": 0.1,
        "mrcnn_bbox_loss": 0.1,
        "mrcnn_mask_loss": 0.1,
    }

    def __init__(self, config: Config):
        """Instantiates an MRCNN model configuration

        :param config: model configuration."""
        self.STEPS_PER_EPOCH = 15 if config.DEBUG else 150
        self.VALIDATION_STEPS = 10 if config.DEBUG else 125
        self.IMAGE_MIN_DIM = config.WIDTH
        self.IMAGE_MAX_DIM = config.WIDTH
        self.NUM_CLASSES = config.NUM_CLASSES

        super().__init__()


class Model:
    def __init__(self, config: Config):
        """Instantiates an MRCNN model

        :param config: model configuration."""
        self.config = config
        self.SHAPE = (config.WIDTH, config.WIDTH)
        self.mrcnn_config = MyMRCNNConfig(config)
        if not self.config.IS_INFERENCE:
            self.init_dataset()
        self.init_model()

    def split_dataframe(self) -> Tuple[List[str], List[str], pd.DataFrame]:
        """Split dataframe 80/20 train/test.

        :return: a tuple containing the list of file paths to the train
           validation images as well as the dataframe containing their
           labels."""
        anns = create_pathology_dataframe(
            self.config.IMAGE_PATH,
            self.config.MASK_PATHS,
        )
        train_names = anns.ID.unique().tolist()  # override with ships

        image_fps_train, image_fps_val = train_test_split(
            train_names, test_size=0.2, random_state=42
        )

        return image_fps_train, image_fps_val, anns

    def init_dataset(self):
        """Constructs the train and validation datasets"""
        image_fps_train, image_fps_val, self.image_annotations = self.split_dataframe()
        class_names = list(self.config.MASK_PATHS.keys())
        annotation_mask_names = list(self.config.MASK_PATHS.keys())
        self.dataset_train = DetectorDataset(
            image_fps_train,
            self.image_annotations,
            self.config.IMAGE_PATH,
            self.SHAPE,
            class_names,
            annotation_mask_names,
            self.config.MASK_COLOR,
        )
        self.dataset_train.prepare()

        self.dataset_val = DetectorDataset(
            image_fps_val,
            self.image_annotations,
            self.config.IMAGE_PATH,
            self.SHAPE,
            class_names,
            annotation_mask_names,
            self.config.MASK_COLOR,
        )
        self.dataset_val.prepare()

    def init_model(self):
        """Loads the model weights. In inference mode, the last layers are
        excluded as transfer learning is performed."""
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)

        self.model = modellib.MaskRCNN(
            mode="inference" if self.config.IS_INFERENCE else "training",
            config=self.mrcnn_config,
            model_dir=self.config.MODEL_DIR,
        )

        if not self.config.IS_INFERENCE:
            # Exclude the last layers because they require a matching
            # number of classes
            self.model.load_weights(
                self.config.WEIGHTS_PATH,
                by_name=True,
                exclude=[
                    "mrcnn_class_logits",
                    "mrcnn_bbox_fc",
                    "mrcnn_bbox",
                    "mrcnn_mask",
                ],
            )
        else:
            self.model.load_weights(self.config.WEIGHTS_PATH, by_name=True)

    def train(self):
        """Starts training the model. It saves the trainning history in
        `self.history` and saves the best model on disk."""
        self.model.train(
            self.dataset_train,
            self.dataset_val,
            learning_rate=self.config.LEARNING_RATE,
            epochs=self.config.EPOCHS,
            layers="heads",
            augmentation=None,
        )
        self.history = self.model.keras_model.history.history
        self.back_up_best_model()

    def get_best_epoch(self) -> Tuple[int, float]:
        """Returns the best epoch and score based on the validation loss.

        :return: a tuple containing the best epoch and its score."""
        best_epoch = np.argmin(self.history["val_loss"])
        score = self.history["val_loss"][best_epoch]
        return best_epoch, score

    def get_best_model_path(self) -> str:
        """Get the path to the best model.

        :return: the path to the best model."""
        model_number = str(self.get_best_epoch()[0]).zfill(4)
        return self.model.log_dir + f"/mask_rcnn_idrid_{model_number}.h5"

    def back_up_best_model(self):
        """Saves the best model. TODO explain model location"""
        best_model_path = Path(self.get_best_model_path())
        target_path = os.path.join(best_model_path.parent.parent, "best_model.h5")
        copyfile(best_model_path, target_path)

    def create_cropped_image(self):
        """Crops the dataset images around the disc / cup region by using the
        model's mask."""
        create_cropped_image(
            self.model,
            self.config.cropped_image.INPUT_PATH_HEALTHY,
            self.config.cropped_image.NAME_HEALTHY,
            self.config.cropped_image.OUTPUT_PATH_HEALTHY,
            self.SHAPE,
        )
        create_cropped_image(
            self.model,
            self.config.cropped_image.INPUT_PATH_GLAUCOMA,
            self.config.cropped_image.NAME_GLAUCOMA,
            self.config.cropped_image.OUTPUT_PATH_GLAUCOMA,
            self.SHAPE,
        )

    def get_iou_score(self) -> float:
        """Calculates the average IOU score over the entire dataset

        :return: iou score."""
        n_masks = len(self.config.MASK_PATHS)
        col_names = list(self.config.MASK_PATHS.keys())
        list_ious = mrcnn_iou_eval(
            self.model, self.image_annotations, n_masks, col_names
        )
        return sum([sum(list_iou) for list_iou in list_ious]) / sum(
            [len(list_iou) for list_iou in list_ious]
        )

    def infer(self, img_path) -> List[Dict[str, Any]]:
        """Calculates the masks for a given image.

        :param img_path: path to the image to calculate the masks for

        :return: a list of dicts, one dict per image. The dict contains:
            - rois: [N, (y1, x1, y2, x2)] detection bounding boxes
            - class_ids: [N] int class IDs
            - scores: [N] float probability scores for the class IDs
            - masks: [H, W, N] instance binary masks"""
        img = imread(img_path)
        img_detect = img.copy()
        return self.model.detect([img_detect], verbose=1)

    def export_dataset_output_dictionary(
        self, export_dir: str, train_paths: List[str], valid_paths: List[str]
    ):
        """Exports two json files (for the trainning and validation sets)
        containing a dictionary with keys being image paths and values being
        the disc to cup ratio.

        :param export_dir: directiory to write json files to
        :param train_paths: a list of paths to the train images
        :valid_paths: a list of paths to the validation images"""
        os.makedirs(export_dir, exist_ok=True)

        for data_files, filename in [
            (train_paths, "train_dic.json"),
            (valid_paths, "valid_dic.json"),
        ]:
            result_dict, failed_images = mmod(
                self.model,
                data_files,
            )
            with open(os.path.join(export_dir, filename), "w") as f:
                json.dump(result_dict, f)
