import os
from dataclasses import dataclass

import numpy as np
from fastai.vision import (ClassificationInterpretation, ImageDataBunch, Path,
                           accuracy, cnn_learner, get_transforms,
                           imagenet_stats, load_learner, models, open_image)
from helpers import SaveBestModel, fmod

freeze_type = {
    "FULLY_UNFROZEN": 0,
    "FULLY_FROZEN": 1,
    "FREEZE_TO": 2,
}


@dataclass
class FreezeConfig:
    """
    Params
    ------
    FREEZE_TYPE:
      0. fully unfrozen
      1. fully frozen
      2. use freeze to
    FREEZE_TO: layers to freeze up to
    """

    FREEZE_TYPE: int = freeze_type["FULLY_UNFROZEN"]
    FREEZE_TO: int = -2


@dataclass
class Config:
    """
    Params
    ------
    freeze: transfer learning model freezing configuration
    TRAIN_DATA_PATH_ROOT: path to train data root
    INFERENCE_DATA_PATH_ROOT: path to folder containing files to infer
    IS_INFERENCE: flag to train the model or infer using it
    MODEL_DIR:
      - if IS_INFERENCE: directory of model to load
      - if not IS_INFERENCE: directory to save model in
    MODEL_NAME: name of model to load when infering
    """
    freeze: FreezeConfig = FreezeConfig()
    TRAIN_DATA_PATH_ROOT: str = ""
    INFERENCE_DATA_PATH_ROOT: str = ""
    IS_INFERENCE: bool = False
    MODEL_DIR: str = ""
    MODEL_NAME: str = "export.pkl"
    EPOCHS: int = 50


class Model:
    def __init__(self, config: Config):
        self.config = config
        self.init_learner()

    def init_learner(self):
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)

        if self.config.IS_INFERENCE:
            self.learner = load_learner(self.config.MODEL_DIR, self.config.MODEL_NAME)
        else:
            self.data = self.load_train_data()
            self.learner = cnn_learner(
                self.data,  # must be a data loader instance (dls) like ImageDataBunch.from_folder()
                models.resnet50,  # pre-trained model chosen
                metrics=accuracy,
                callback_fns=SaveBestModel,
                path=self.config.MODEL_DIR,
            )
            self.setup_frozen_model()
            self.interpretation = ClassificationInterpretation.from_learner(
                self.learner
            )

    def setup_frozen_model(self):
        if self.config.freeze.FREEZE_TYPE == freeze_type["FULLY_UNFROZEN"]:
            self.learner.unfreeze()
        elif self.config.freeze.FREEZE_TYPE == freeze_type["FULLY_FROZEN"]:
            self.learner.freeze()
        else:
            self.learner.freeze_to(self.config.freeze.FREEZE_TO)

    def load_inference_data(self):
        list_img = os.listdir(self.config.INFERENCE_DATA_PATH_ROOT)
        list_paths = []
        for path in list_img:
            path = os.path.join(self.config.INFERENCE_DATA_PATH_ROOT, path)
            list_paths.append(path)
        return list_paths

    def load_train_data(self):
        # We create a DataBunch object from the Data folder
        return ImageDataBunch.from_folder(
            Path(self.config.TRAIN_DATA_PATH_ROOT),
            train="train",
            valid="valid",
            # train='.',
            # valid_pct=0.2, # ratio split for train & test
            ds_tfms=get_transforms(  # Dataset transformations (augmentation),
                do_flip=False,  # generates copies of the images. Here,
                flip_vert=False,  # no transformation applied.
                max_rotate=0,
                p_affine=0,
            ),
            size=(256, 256),  # size of the images created
            num_workers=4,
            bs=16,
        ).normalize(imagenet_stats)

    def get_results(self):
        if self.config.IS_INFERENCE:
            return fmod(self.learner, self.load_inference_data())
        else:
            return fmod(self.learner, self.data.valid_ds.items)

    def infer(self, img_path: str):
        return self.learner.predict(open_image(img_path))

    def train(self):
        self.model.fit_one_cycle(self.config.EPOCHS)

