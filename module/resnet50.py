import json
import os
from dataclasses import dataclass
from typing import List, Dict

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
    :param FREEZE_TYPE: How are the model layers to be freezed
        0. fully unfrozen
        1. fully frozen
        2. use freeze to
    FREEZE_TO: layers to freeze up to. The integer `-2` will freeze the last
        two layers of the model
    """
    FREEZE_TYPE: int = freeze_type["FULLY_UNFROZEN"]
    FREEZE_TO: int = -2


@dataclass
class Config:
    """
    :param freeze: transfer learning model freezing configuration
    :param TRAIN_DATA_PATH_ROOT: path to train data root
    :param INFERENCE_DATA_PATH_ROOT: path to folder containing files to infer
    :param IS_INFERENCE: flag to train the model or infer using it
    :param MODEL_DIR:
      - if IS_INFERENCE: directory of model to load
      - if not IS_INFERENCE: directory to save model in
    :param MODEL_NAME: name of model to load when infering
    :param EPOCHS: number of epochs to train for
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
        """Instantiates an Resnet50 model

        :param config: model configuration."""
        self.config = config
        self.init_learner()

    def init_learner(self):
        """"Initializes the learner based on whether or not inference
        is performed."""
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
        """Freezes the layers based of the freeze configuration"""
        if self.config.freeze.FREEZE_TYPE == freeze_type["FULLY_UNFROZEN"]:
            self.learner.unfreeze()
        elif self.config.freeze.FREEZE_TYPE == freeze_type["FULLY_FROZEN"]:
            self.learner.freeze()
        else:
            self.learner.freeze_to(self.config.freeze.FREEZE_TO)

    def load_inference_data(self) -> List[str]:
        """Constructs a list of paths for the inference data.

        :return: a list of paths to the images to infer."""
        list_img = os.listdir(self.config.INFERENCE_DATA_PATH_ROOT)
        list_paths = []
        for path in list_img:
            path = os.path.join(self.config.INFERENCE_DATA_PATH_ROOT, path)
            list_paths.append(path)
        return list_paths

    def load_train_data(self) -> ImageDataBunch:
        """Creates a DataBunch object from the Data folder.

        :return: an ImageDataBunch with the train data."""
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

    def get_results(self) -> Dict[str, float]:
        """Stores the results in a dictionary. If the model is in
        inference mode, the results are calculated on the images to
        infer. Otherwise, it is the images from the validation dataset
        that are infered and their glaucoma scores that are stored in the
        dictionary.

        :return: a dictionary with the images name as the key and the
            images Glaucoma score as the associated value"""
        if self.config.IS_INFERENCE:
            return fmod(self.learner, self.load_inference_data())
        else:
            return fmod(self.learner, self.data.valid_ds.items)

    def export_dataset_output_dictionary(self, export_dir: str):
        """Exports two json files (for the trainning and validation sets)
        containing a dictionary with keys being image paths and values being
        the glaucoma scores.

        :param export_dir: directiory to write json files to"""
        os.makedirs(export_dir, exist_ok=True)
        if self.config.TRAIN_DATA_PATH_ROOT:
            data = self.load_train_data()
            for dataset, filename in [
                (data.train_ds, "train_dic.json"),
                (data.valid_ds, "valid_dic.json"),
            ]:
                result_dict = fmod(self.learner, dataset.items)
                with open(os.path.join(export_dir, filename), "w") as f:
                    json.dump(result_dict, f)
        else:
            print("Please set the TRAIN_DATA_PATH_ROOT configuration attibute.")

    def infer(self, img_path: str) -> Tuple[int, int, List[float]]:
        """Calculates the glaucoma score for an input image.

        :param img_path: path to the image to calculate the glaucoma for
        :return: the predicted class, label and probabilities for `img_path`"""
        return self.learner.predict(open_image(img_path))

    def train(self):
        """Starts training the model."""
        self.learner.fit_one_cycle(self.config.EPOCHS)
