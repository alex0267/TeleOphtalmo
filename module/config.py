import helpers
import os
import resnet50
import MRCNN
import logistic_regression


HOME = "/home/jupyter"
N_EPOCHS = 2
USE_PRODUCTION_MODEL = True


class Branch1:
    def train(self):
        return resnet50.Config(
            resnet50.FreezeConfig(
                FREEZE_TYPE=resnet50.freeze_type["FREEZE_TO"],
                FREEZE_TO=-2,
            ),
            TRAIN_DATA_PATH_ROOT="/app/datasets/ORIGA",
            MODEL_DIR="/app/models/branch1",
            EPOCHS=N_EPOCHS,
        )

    def infer(self):
        config = self.train()
        config.IS_INFERENCE = True
        config.INFERENCE_DATA_PATH_ROOT = "/app/datasets/ORIGA/valid/Glaucoma"
        if USE_PRODUCTION_MODEL:
            config.MODEL_DIR = "/app/models/final/1st_branch"
            config.MODEL_NAME = "export.pkl"
        else:
            config.MODEL_NAME = "best_model.pkl"
        return config


class Branch2:
    def train(self):
        return resnet50.Config(
            resnet50.FreezeConfig(
                FREEZE_TYPE=resnet50.freeze_type["FREEZE_TO"],
                FREEZE_TO=-2,
            ),
            TRAIN_DATA_PATH_ROOT="/app/datasets/ORIGA_cropped",
            MODEL_DIR="/app/models/branch2",
            EPOCHS=N_EPOCHS,
        )

    def infer(self):
        config = self.train()
        config.INFERENCE_DATA_PATH_ROOT = "/app/datasets/ORIGA/valid"
        config.IS_INFERENCE = True
        if USE_PRODUCTION_MODEL:
            config.MODEL_DIR = "/app/models/final/2d_branch"
            config.MODEL_NAME = "export.pkl"
        else:
            config.MODEL_NAME = "best_model.pkl"
        return config


class LogReg:
    def train_3b(self):
        HOME = "/app"
        return logistic_regression.Config(
            PATH_1_TRAIN=os.path.join(HOME, "output", "b1", "train_dic.json"),
            PATH_1_VAL=os.path.join(HOME, "output", "b1", "valid_dic.json"),
            PATH_2_TRAIN=os.path.join(HOME, "output", "b2", "train_dic.json"),
            PATH_2_VAL=os.path.join(HOME, "output", "b2", "valid_dic.json"),
            PATH_3_TRAIN=os.path.join(HOME, "output", "b3", "train_dic.json"),
            PATH_3_VAL=os.path.join(HOME, "output", "b3", "valid_dic.json"),
            N_BRANCHES=3,
            MODEL_PATH=os.path.join(HOME, "models", "logreg", "classifier_3b.sav"),
            IS_INFERENCE=False,
        )

    def infer_3b(self):
        config = self.train_3b()
        config.IS_INFERENCE = True
        return config

    def train_2b(self):
        HOME = "/app"
        config = self.train_3b()
        config.N_BRANCHES = 2
        config.MODEL_PATH = os.path.join(HOME, "models", "logreg", "classifier_2b.sav")
        return config

    def infer_2b(self):
        config = self.train_2b()
        config.IS_INFERENCE = True
        return config


class Cropper:
    def train(self):
        DATA_DIR = "/app/datasets/IDRID"
        cropped_image_config = MRCNN.CroppedImageConfig(
            INPUT_PATH_GLAUCOMA="/app/datasets/ORIGA/Glaucoma/",
            INPUT_PATH_HEALTHY="/app/datasets/ORIGA/Healthy/",
            NAME_GLAUCOMA="glaucoma",
            NAME_HEALTHY="healthy",
            OUTPUT_PATH_GLAUCOMA="/app/datasets/ORIGA_cropped/glaucoma/",
            OUTPUT_PATH_HEALTHY="/app/datasets/ORIGA_cropped/healthy/",
        )
        return MRCNN.Config(
            IS_INFERENCE=False,
            USE_GPU=True,
            DEBUG=True,
            WIDTH=1024,
            NUM_CLASSES=2,
            MASK_COLOR=helpers.COLORS["red"],
            MASK_PATHS={
                "Disc": os.path.join(
                    DATA_DIR,
                    "A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/5. Optic Disc",
                ),
            },
            IMAGE_PATH=os.path.join(
                DATA_DIR, "A. Segmentation/1. Original Images/a. Training Set/"
            ),
            WEIGHTS_PATH="/app/mask_rcnn_coco.h5",
            MODEL_DIR="/app/models/branch2/",
            LEARNING_RATE=0.0001,
            EPOCHS=N_EPOCHS,
            cropped_image=cropped_image_config,
        )

    def infer(self):
        config = self.train()
        config.IS_INFERENCE=True
        if USE_PRODUCTION_MODEL:
            config.WEIGHTS_PATH = "/app/models/final/mrcnn_b2.h5"
        else:
            config.WEIGHTS_PATH = "/app/models/branch2/best_model.h5"
        return config


class Ratio:
    def train(self):
        DATA_DIR = "/app/datasets/MAGRABIA"
        return MRCNN.Config(
            IS_INFERENCE=False,
            USE_GPU=True,
            DEBUG=True,
            WIDTH=1024,
            NUM_CLASSES=3,
            MASK_COLOR=helpers.COLORS["blue"],
            MASK_PATHS={
                "Disc": os.path.join(
                    DATA_DIR,
                    "disc_segmented",
                ),
                "Cup": os.path.join(
                    DATA_DIR,
                    "cup_segmented",
                ),
            },
            IMAGE_PATH=os.path.join(DATA_DIR, "original"),
            WEIGHTS_PATH=f"/app/mask_rcnn_coco.h5",
            MODEL_DIR="/app/models/branch3/",
            LEARNING_RATE=0.0001,
            EPOCHS=N_EPOCHS,
        )

    def infer(self):
        config = self.train()
        config.IS_INFERENCE = True
        if USE_PRODUCTION_MODEL:
            config.WEIGHTS_PATH = "/app/models/final/mrcnn_b3.h5"
        else:
            config.WEIGHTS_PATH = "app/models/branch3/best_model.h5"
        return config
