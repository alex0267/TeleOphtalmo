import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from shutil import copyfile
from typing import Dict

import mrcnn.model as modellib
import numpy as np
import tensorflow as tf
from helpers import (COLORS, DetectorDataset, create_cropped_image,
                     create_pathology_dataframe, mmod, mrcnn_iou_eval)
from keras.backend.tensorflow_backend import set_session
from mrcnn.config import Config as MRCNNConfig
from skimage.io import imread
from sklearn.model_selection import train_test_split


@dataclass
class CroppedImageConfig:
    INPUT_PATH_GLAUCOMA: str = ""
    INPUT_PATH_HEALTHY: str = ""
    NAME_GLAUCOMA: str = ""
    NAME_HEALTHY: str = ""
    OUTPUT_PATH_GLAUCOMA: str = ""
    OUTPUT_PATH_HEALTHY: str = ""


@dataclass
class Config:
    IS_INFERENCE: bool = False
    MODEL_PATH: str = ""
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
        self.STEPS_PER_EPOCH = 15 if config.DEBUG else 150
        self.VALIDATION_STEPS = 10 if config.DEBUG else 125
        self.IMAGE_MIN_DIM = config.WIDTH
        self.IMAGE_MAX_DIM = config.WIDTH
        self.NUM_CLASSES = config.NUM_CLASSES

        super().__init__()


class Model:
    def __init__(self, config: Config):
        self.config = config
        self.SHAPE = (config.WIDTH, config.WIDTH)
        self.mrcnn_config = MyMRCNNConfig(config)
        self.setup_gpu()
        if not self.config.IS_INFERENCE:
            self.init_dataset()
        self.init_model()

    def setup_gpu(self):
        if self.config.USE_GPU:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.log_device_placement = True
            sess = tf.Session(config=config)
            set_session(sess)

    def split_dataframe(self):
        """Split dataframe 80/20 train/test"""
        anns = create_pathology_dataframe(
            self.config.IMAGE_PATH,
            self.config.MASK_PATHS,
        )
        train_names = anns.ID.unique().tolist()  # override with ships

        test_size = (
            self.mrcnn_config.VALIDATION_STEPS * self.mrcnn_config.IMAGES_PER_GPU
        )
        image_fps_train, image_fps_val = train_test_split(
            train_names, test_size=0.2, random_state=42
        )

        return image_fps_train, image_fps_val, anns

    def init_dataset(self):
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

    def get_best_epoch(self):
        best_epoch = np.argmin(self.history["val_loss"])
        score = self.history["val_loss"][best_epoch]
        return best_epoch, score

    def get_best_model_path(self):
        model_number = str(self.get_best_epoch()[0]).zfill(4)
        return self.model.log_dir + f"/mask_rcnn_idrid_{model_number}.h5"

    def back_up_best_model(self):
        best_model_path = Path(self.get_best_model_path())
        target_path = os.path.join(best_model_path.parent.parent, "best_model.h5")
        copyfile(best_model_path, target_path)

    def create_cropped_image(self):
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

    def get_iou_score(self):
        n_masks = len(self.config.MASK_PATHS)
        col_names = list(self.config.MASK_PATHS.keys())
        list_ious = mrcnn_iou_eval(
            self.model, self.image_annotations, n_masks, col_names
        )
        return sum([sum(list_iou) for list_iou in list_ious]) / sum(
            [len(list_iou) for list_iou in list_ious]
        )

    def infer(self, img_path):
        img = imread(img_path)
        img_detect = img.copy()
        return self.model.detect([img_detect], verbose=1)

    def export_dataset_output_dictionary(self, export_path: str):
        train, val, anns = self.split_dataframe()
        for dataset, filename in [
            (train, "train_dic.json"),
            (val, "valid_dic.json"),
        ]:
            result_dict = mmod(
                self.model,
                [
                    os.path.join(self.config.IMAGE_PATH, filename)
                    for filename in dataset
                ],
            )
            with open(os.path.join(export_path, filename), "w") as f:
                json.dump(result_dict, f)
