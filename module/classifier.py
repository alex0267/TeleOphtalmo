import glob
import os
from dataclasses import dataclass
from shutil import copyfile

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

    def make_logreg_dataset(self):
        HOME = "/home/thomas/TeleOphtalmo/module/"
        self.branch1.export_dataset_output_dictionary(os.path.join(HOME, "output/b1"))
        self.branch2.export_dataset_output_dictionary(os.path.join(HOME, "output/b2"))

        train_paths = glob.glob("/home/jupyter/Data/train/Glaucoma/*")
        train_paths += glob.glob("/home/jupyter/Data/train/Healthy/*")
        valid_paths = glob.glob("/home/jupyter/Data/valid/Glaucoma/*")
        valid_paths += glob.glob("/home/jupyter/Data/valid/Healthy/*")
        export_path = os.path.join(HOME, "output/b3")
        self.ratio.export_dataset_output_dictionary(
            export_path, train_paths, valid_paths
        )

    def train_feature_engineering(self):
        self.cropper.train()
        self.ratio.train()

    def train(self):
        self.branch1.train()
        self.branch2.train()
        self.make_logreg_dataset()
        self.logreg.train()

    def crop_image(self, img_path):
        # move image to /tmp
        base_name = os.path.basename(img_path)
        tmp_img_path = os.path.join("/tmp", base_name)
        copyfile(img_path, tmp_img_path)

        cropped_image_path = helpers.create_cropped_image(
            self.cropper, "/tmp", base_name, "/tmp", self.cropper.SHAPE
        )

        # TODO remove tmp_img_path
        if len(cropped_image_path) == 1:
            return cropped_image_path[0]
        else:
            return None

    def infer(self, img_path):
        cropped_img_path = self.cropper.infer(img_path)
        if cropped_img_path:
            results_branch1 = self.branch1.infer(img_path)
            results_branch2 = self.branch2.infer(cropped_img_path)
            success, ratio = helpers.cup_to_disc_ratio(self.ratio, img_path)
            if success:
                X = [[results_branch1, results_branch2, ratio]]
            else:
                X = [[results_branch1, results_branch2]]
            return self.logreg.infer(X)
        else:
            return None
