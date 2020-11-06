import os

import MRCNN
import resnet50

HOME = "/home/jupyter"


class Branch1:
    def __init__(self):
        pass

    def train(self):
        config = resnet50.Config(
            resnet50.FreezeConfig(
                FREEZE_TYPE=resnet50.freeze_type["FREEZE_TO"],
                FREEZE_TO=-2,
            ),
            f"{HOME}/Data/",
        )
        model = Model(config)
        model.learner.fit_one_cycle(50)
        return model

    def infer(self):
        config = resnet50.Config(
            INFERENCE_DATA_PATH_ROOT="/home/jupyter/Data/valid",
            IS_INFERENCE=True,
            MODEL_PATH="",
        )
        infer_model = Model(config)
        return infer_model.get_results()


class Branch2:
    def __init__(self):
        pass

    def train_mrcnn(self):
        DATA_DIR = f"{HOME}/Second_branch/data_train_mrcnn/"
        config = MRCNN.Config(
            IS_INFERENCE=False,
            USE_GPU=True,
            DEBUG=True,
            WIDTH=1024,
            NUM_CLASSES=2,
            MASK_PATH=os.path.join(
                DATA_DIR,
                "A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/",
            ),
            IMAGE_PATH=os.path.join(
                DATA_DIR, "A. Segmentation/1. Original Images/a. Training Set/"
            ),
            WEIGHTS_PATH=f"{HOME}/mask_rcnn_coco.h5",
            ROOT_DIR="Second_branch/",
            LEARNING_RATE=0.0001,
        )
        model = Model(config)
        model.train()
        return model

    def crop_images(self):
        self.train_mrcnn().create_cropped_image()

    def train_resnet50(self):
        config = resnet50.Config(
            resnet50.FreezeConfig(
                FREEZE_TYPE=resnet50.freeze_type["FREEZE_TO"],
                FREEZE_TO=-2,
            ),
            f"{HOME}/Second_branch/output_MaskRcnn_ORIGA/",
        )
        model = Model(config)
        model.learner.fit_one_cycle(50)
        return model

    def infer(self):
        config = resnet50.Config(
            INFERENCE_DATA_PATH_ROOT="/home/jupyter/Data/valid",
            IS_INFERENCE=True,
            MODEL_PATH="best_model_colab",
        )
        model = Model(config)
        return model.get_results()


class Model:
    def __init__(self):
        self.branch1 = Branch1()
        self.branch2 = Branch2()

    def train(self):
        self.branch1.train()
        self.branch2.train_mrcnn().create_cropped_image()
        self.branch2.train_resnet50()

    def infer(self):
        results_branch1 = self.branch1.infer()
        results_branch2 = self.branch2.infer()
        return [results_branch1, results_branch2]
