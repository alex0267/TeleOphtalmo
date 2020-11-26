import glob
import os
from dataclasses import dataclass
from shutil import copyfile
import shutil
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import helpers
import logistic_regression
import MRCNN
import resnet50
import config

HOME = "/home/jupyter"
N_EPOCHS = 2


@dataclass
class Config:
    branch1: resnet50.Config
    branch2: resnet50.Config
    cropper: MRCNN.Config
    ratio: MRCNN.Config
    logreg_3b: logistic_regression.Config
    logreg_2b: logistic_regression.Config


train_config = Config(
    branch1=config.Branch1().train(),
    branch2=config.Branch2().train(),
    cropper=config.Cropper().train(),
    ratio=config.Ratio().train(),
    logreg_3b=config.LogReg().train_3b(),
    logreg_2b=config.LogReg().train_2b(),
)
infer_config = Config(
    branch1=config.Branch1().infer(),
    branch2=config.Branch2().infer(),
    cropper=config.Cropper().infer(),
    ratio=config.Ratio().infer(),
    logreg_3b=config.LogReg().infer_3b(),
    logreg_2b=config.LogReg().infer_2b(),
)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)


class Model:
    def __init__(self, config: Config):
        self.branch1 = resnet50.Model(config.branch1)
        self.branch2 = resnet50.Model(config.branch2)
        self.ratio = MRCNN.Model(config.ratio)
        self.cropper = MRCNN.Model(config.cropper)
        self.logreg_3b = logistic_regression.Model(config.logreg_3b)
        self.logreg_2b = logistic_regression.Model(config.logreg_2b)

    def export_branch2_dataset(self):
        self.cropper.create_cropped_image()

        # TODO clean split folders

        helpers.train_valid_split(
            self.branch2.config.TRAIN_DATA_PATH_ROOT,
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
        self.logreg_2b.train()
        self.logreg_3b.train()

    def crop_image(self, img_path):
        # move image to /tmp
        base_name = os.path.basename(img_path)

        tmp_dir = "/tmp/teleophtalmo"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)

        tmp_img_path = os.path.join(tmp_dir, base_name)
        copyfile(img_path, tmp_img_path)

        cropped_image_path = helpers.create_cropped_image(
            self.cropper.model, tmp_dir, base_name, tmp_dir, self.cropper.SHAPE
        )

        if len(cropped_image_path) == 1:
            return cropped_image_path[0]
        else:
            return None

    def infer(self, img_path):
        cropped_img_path = self.crop_image(img_path)
        if cropped_img_path:
            results_branch1 = self.branch1.infer(img_path)
            results_branch2 = self.branch2.infer(cropped_img_path)
            success, ratio = helpers.cup_to_disc_ratio(self.ratio.model, img_path)
            if success:
                print("===> 3b")
                X = [[results_branch1[2][0], results_branch2[2][0], ratio]]
                return self.logreg_3b.model.predict(X), self.logreg_3b.model.predict_proba(X)
            else:
                print("===> 2b")
                X = [[results_branch1[2][0], results_branch2[2][0]]]
                return self.logreg_2b.model.predict(X), self.logreg_2b.model.predict_proba(X)
        else:
            return None

    def score(self, data_dir="/home/jupyter/Data/valid/"):
        path_healthy = os.listdir(os.path.join(data_dir, "Healthy"))
        path_glaucoma = os.listdir(os.path.join(data_dir, "Glaucoma"))

        healthy_class_id = 0
        glaucoma_class_id = 1

        scores_healthy = []
        for file_name in path_healthy:
            print(scores_healthy)
            if file_name == '.ipynb_checkpoints':
                continue
            else:
                file_path = os.path.join(data_dir, "Healthy", file_name)
                class_id = self.infer(file_path)[0][0]
                if class_id == healthy_class_id:
                    scores_healthy.append(1)
                else:
                    scores_healthy.append(0)
        score_healthy = sum(scores_healthy) / len(scores_healthy)

        scores_glaucoma = []
        for file_name in path_glaucoma:
            print(scores_glaucoma)
            if file_name == '.ipynb_checkpoints':
                continue
            else:
                file_path = os.path.join(data_dir, "Glaucoma", file_name)
                class_id = self.infer(file_path)[0][0]
                if class_id == glaucoma_class_id:
                    scores_glaucoma.append(1)
                else:
                    scores_glaucoma.append(0)
        score_glaucoma = sum(scores_glaucoma) / len(scores_glaucoma)

        return {
            "score_healthy": score_healthy,
            "score_glaucoma": score_glaucoma,
        }


