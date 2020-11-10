import os
from dataclasses import dataclass, field
from typing import Dict

import mrcnn.model as modellib
import numpy as np
import tensorflow as tf
from helpers import (DetectorDataset, create_cropped_image,
                     create_pathology_dataframe, mrcnn_iou_eval)
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
    IMAGE_PATH: str = ""
    WEIGHTS_PATH: str = ""
    MODEL_DIR: str = ""
    LEARNING_RATE: float = 0.0001
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
        )
        self.dataset_train.prepare()

        self.dataset_val = DetectorDataset(
            image_fps_val,
            self.image_annotations,
            self.config.IMAGE_PATH,
            self.SHAPE,
            class_names,
            annotation_mask_names,
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
            epochs=4 if self.config.DEBUG else 250,
            layers="heads",
            augmentation=None,
        )
        self.history = self.model.keras_model.history.history

    def get_best_epoch(self):
        best_epoch = np.argmin(self.history["val_loss"])
        score = self.history["val_loss"][best_epoch]
        return best_epoch, score

    def get_best_model_path(self):
        model_number = str(self.get_best_epoch()[0]).zfill(4)
        return self.model.log_dir + f"/mask_rcnn_idrid_{model_number}.h5"

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
        list_iou = mrcnn_iou_eval(self.model, self.image_annotations)
        return sum(list_iou) / len(list_iou)

    def infer(self, img_path):
        img = imread(img_path)
        img_detect = img.copy()
        return self.model.detect([img_detect], verbose=1)


if __name__ == "__main__":
    # Train
    DATA_DIR = "/home/jupyter/Second_branch/data_train_mrcnn/"
    cropped_image_config = CroppedImageConfig(
        INPUT_PATH_GLAUCOMA="/home/jupyter/Data/Glaucoma/",
        INPUT_PATH_HEALTHY="/home/jupyter/Data/Healthy/",
        NAME_GLAUCOMA="glaucoma",
        NAME_HEALTHY="healthy",
        OUTPUT_PATH_GLAUCOMA="/home/thomas/TeleOphtalmo/module/output_MaskRcnn_ORIGA/glaucoma/",
        OUTPUT_PATH_HEALTHY="/home/thomas/TeleOphtalmo/module/output_MaskRcnn_ORIGA/healthy/",
    )
    config = Config(
        IS_INFERENCE=False,
        USE_GPU=True,
        DEBUG=True,
        WIDTH=1024,
        NUM_CLASSES=2,
        MASK_PATHS={
            "Disc": os.path.join(
                DATA_DIR,
                "A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/5. Optic Disc",
            ),
        },
        IMAGE_PATH=os.path.join(
            DATA_DIR, "A. Segmentation/1. Original Images/a. Training Set/"
        ),
        WEIGHTS_PATH="/home/jupyter/mask_rcnn_coco.h5",
        MODEL_DIR="/home/thomas/TeleOphtalmo/module/models/branch2/",
        LEARNING_RATE=0.0001,
        cropped_image=cropped_image_config,
    )
    model = Model(config)
    model.train()
    best_model = model.get_best_model_path()

    # Infer
    config.IS_INFERENCE = True
    config.WEIGHTS_PATH = best_model
    model = Model(config)
    model.create_cropped_image()
