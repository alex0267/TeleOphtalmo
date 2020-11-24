import os
from dataclasses import dataclass

import feature_engineering
import helpers
import logistic_regression
import MRCNN
import resnet50

HOME = "/home/jupyter"
N_EPOCHS = 2


@dataclass
class Config:
    branch1: resnet50.Config
    branch2: resnet50.Config
    cropper: MRCNN.Config
    ratio: MRCNN.Config
    logreg: logistic_regression.Config


class Model:
    def __init__(self, config: Config):
        self.branch1 = resnet50.Model(config.branch1)
        self.branch2 = resnet50.Model(config.branch2)
        self.cropper = MRCNN.Model(config.cropper)
        self.ratio = MRCNN.Model(config.ratio)
        self.logreg = logistic_regression.Model(config.logreg)

    def export_branch2_dataset(self):
        self.cropper.crop_images()
        helpers.train_valid_split(
            "/home/thomas/TeleOphtalmo/module/output_MaskRcnn_ORIGA",
            "healthy",
            "glaucoma",
        )

    def train_feature_engineering(self):
        self.cropper.train()
        self.ratio.train()

    def make_train_dataset(self):
        self.cropper.infer()
        self.ratio.infer()

    def train(self):
        self.branch1.train()
        self.export_branch2_dataset()
        self.branch2.train()
        self.logreg.train()

    def export_logreg_dataset(self):
        HOME = "/home/thomas/TeleOphtalmo/module/"
        self.branch1.export_dataset_output_dictionary(HOME)
        self.branch2.export_dataset_output_dictionary(HOME)
        self.ratio.export_dataset_output_dictionary(HOME)

    def infer(self, img_path):
        results_branch1 = self.branch1.infer(img_path)
        cropped_img_path = self.cropper.infer(img_path)
        results_branch2 = self.branch2.infer(cropped_img_path)
        ratio = self.ratio.infer(img_path)
        X = [[results_branch1, results_branch2, ratio]]
        return self.logreg.infer(X)
