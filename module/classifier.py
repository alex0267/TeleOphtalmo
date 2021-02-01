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
from typing import Optional, Tuple, List

HOME = "/home/jupyter"
N_EPOCHS = 2


@dataclass
class Config:
    """Classifier configuration.

    :param branch1: branch1 configuration
    :param branch2: branch2 configuration
    :param cropper: cropper configuration
    :param ratio: ratio configuration
    :param logreg_3b: 3 branch logistic regression configuration
    :param logreg_2b: 2 branch logistic regression configuration"""
    branch1: resnet50.Config
    branch2: resnet50.Config
    cropper: MRCNN.Config
    ratio: MRCNN.Config
    logreg_3b: logistic_regression.Config
    logreg_2b: logistic_regression.Config


@dataclass
class AverageClassificationScore:
    """The average classification score over a dataset for the healthy
    retinas and the ones containing a glaucoma.

    :param healthy: the average classification score over healthy retinas
    :param glaucoma: the average classification score over healthy glaucoma
    :param failed_images: list of path to images that could not be classified
    """
    healthy: float
    glaucoma: float
    failed_images: List[str]


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
        """Instanciate a classifier.

        :param config: classifier configuration"""
        self.config = config

    @property
    def branch1(self) -> resnet50.Model:
        """The branch1 model, used to classify the retina ORIGA raw images.

        :return: the branch1 model"""
        if not hasattr(self, '_branch1'):
            self._branch1 = resnet50.Model(self.config.branch1)

        return self._branch1

    @property
    def branch2(self) -> resnet50.Model:
        """The branch2 model, used to classify the retina ORIGA images
        cropped around the cup.

        :return: the branch2 model"""
        if not hasattr(self, '_branch2'):
            self._branch2 = resnet50.Model(self.config.branch2)

        return self._branch2

    @property
    def ratio(self) -> MRCNN.Model:
        """The model used to compute the disc/cup ratio on the ORIGA dataset.

        :return: the ratio model"""
        if not hasattr(self, '_ratio'):
            self._ratio = MRCNN.Model(self.config.ratio)

        return self._ratio

    @property
    def cropper(self) -> MRCNN.Model:
        """The model used to crop ORIGA images around the cup.

        :return: the cropper model"""
        if not hasattr(self, '_cropper'):
            self._cropper = MRCNN.Model(self.config.cropper)

        return self._cropper

    @property
    def logreg_2b(self) -> logistic_regression.Model:
        """The logistic regression used to classify the retina using only
        the branch1 and branch2 models.

        :return: the 2 branch logistic regression model"""
        if not hasattr(self, '_logreg_2b'):
            self._logreg_2b = logistic_regression.Model(self.config.logreg_2b)

        return self._logreg_2b

    @property
    def logreg_3b(self) -> logistic_regression.Model:
        """The logistic regression used to classify the retina using
        the branch1 model, branch2 model and dics/cup ratio.

        :return: the 3 branch logistic regression model"""
        if not hasattr(self, '_logreg_3b'):
            self._logreg_3b = logistic_regression.Model(self.config.logreg_3b)

        return self._logreg_3b

    def export_branch2_dataset(self):
        """Export the dataset used for training the second branch. That is,
        crop the ORIGA dataset around the disc."""
        self.cropper.create_cropped_image()

        helpers.train_valid_split(
            self.branch2.config.TRAIN_DATA_PATH_ROOT,
            "healthy",
            "glaucoma",
        )

    def make_logreg_dataset(self):
        """Export the dataset used to train the logistic regression. That is,
        export the classifications from the branches 1 & 2 as well as the
        disc/cup ratios for the ORIGA dataset."""
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
        """Train the models used for feature engineering. That is, train the
        models used to crop the ORIGA dataset around the cup and calculate
        the disc/cup ratios."""
        self.cropper.train()
        self.ratio.train()

    def train_branches(self):
        """Train the models used to classify images from the ORIGA datasets."""
        self.branch1.train()
        self.branch2.train()

    def train_logreg(self):
        """Train the logistic regressions using:
            - case 1: only the classifications from the 2 branches.
            - case 2: the classifications from the 2 branches as well as
                      the disc/cup raio."""
        self.logreg_2b.train()
        self.logreg_3b.train()

    def crop_image(self, img_path: str) -> Optional[str]:
        """Crop the input image around the cup.

        :param img_path: path to the image to crop
        :return: the path to the cropped image if the cropping was successful
            or `None` otherwise"""
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

    def infer(self, img_path: str) -> Optional[Tuple[List[int], float]]:
        """Classify an retina using only the 2 branch logistic regression
        if the disc/cup ratio could not be computed and the 3 branch logistic
        regression otherwise.

        :param img_path: path to the image to classify.
        :return: a tuple containing the classification as first element and
            the probability of glaucoma as second element if the input image
            could successfully be cropped and `None` otherwise."""
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

    def score(self, data_dir="/home/jupyter/Data/valid/") -> AverageClassificationScore:
        """Score the classifier on all the retina images contained in an
        input folder.

        :param data_dir: path to the directory containing the images to
            score the classifier over.
        :return: the average classification score for healthy and glaucoma
            images."""
        path_healthy = os.listdir(os.path.join(data_dir, "Healthy"))
        path_glaucoma = os.listdir(os.path.join(data_dir, "Glaucoma"))

        healthy_class_id = 0
        glaucoma_class_id = 1

        failed_images: List[str] = []
        scores_healthy: List[int] = []
        for file_name in path_healthy:
            print(scores_healthy)
            if file_name == '.ipynb_checkpoints':
                continue
            else:
                file_path = os.path.join(data_dir, "Healthy", file_name)
                inference_result = self.infer(file_path)
                if inference_result:
                    class_id = inference_result[0][0]
                    if class_id == healthy_class_id:
                        scores_healthy.append(1)
                    else:
                        scores_healthy.append(0)
                else:
                    failed_images.append(file_path)
        score_healthy = sum(scores_healthy) / len(scores_healthy)

        scores_glaucoma: List[int] = []
        for file_name in path_glaucoma:
            print(scores_glaucoma)
            if file_name == '.ipynb_checkpoints':
                continue
            else:
                file_path = os.path.join(data_dir, "Glaucoma", file_name)
                inference_result = self.infer(file_path)
                if inference_result:
                    class_id = inference_result[0][0]
                    if class_id == glaucoma_class_id:
                        scores_glaucoma.append(1)
                    else:
                        scores_glaucoma.append(0)
                else:
                    failed_images.append(file_path)
        score_glaucoma = sum(scores_glaucoma) / len(scores_glaucoma)

        return AverageClassificationScore(score_healthy, score_glaucoma, failed_images)


if __name__ == "__main__":
    train_model = Model(train_config)
    infer_model = Model(infer_config)

    train_model.train_feature_engineering()
    infer_model.export_branch2_dataset()
    train_model.train_branches()
    infer_model.make_logreg_dataset()
    train_model.train_logreg()
    infer_model.score()
