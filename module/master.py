import os

import helpers
import MRCNN
import resnet50

HOME = "/home/jupyter"
N_EPOCHS = 2


class Branch1:
    def __init__(self):
        pass

    def train(self):
        config = resnet50.Config(
            resnet50.FreezeConfig(
                FREEZE_TYPE=resnet50.freeze_type["FREEZE_TO"],
                FREEZE_TO=-2,
            ),
            TRAIN_DATA_PATH_ROOT=f"{HOME}/Data/",
            MODEL_DIR="/home/thomas/TeleOphtalmo/module/models/branch1",
            EPOCHS=N_EPOCHS,
        )
        model = resnet50.Model(config)
        model.train()
        return model

    def infer(self):
        config = resnet50.Config(
            INFERENCE_DATA_PATH_ROOT="/home/jupyter/Data/valid/Glaucoma",
            IS_INFERENCE=True,
            MODEL_DIR="/home/thomas/TeleOphtalmo/module/models/branch1",
            MODEL_NAME="best_model.pkl",
        )
        return resnet50.Model(config).get_results()


class Branch2:
    def __init__(self):
        pass

    def train_mrcnn(self):
        DATA_DIR = f"{HOME}/Second_branch/data_train_mrcnn/"
        cropped_image_config = MRCNN.CroppedImageConfig(
            INPUT_PATH_GLAUCOMA="/home/jupyter/Data/Glaucoma/",
            INPUT_PATH_HEALTHY="/home/jupyter/Data/Healthy/",
            NAME_GLAUCOMA="glaucoma",
            NAME_HEALTHY="healthy",
            OUTPUT_PATH_GLAUCOMA="/home/thomas/TeleOphtalmo/module/output_MaskRcnn_ORIGA/glaucoma/",
            OUTPUT_PATH_HEALTHY="/home/thomas/TeleOphtalmo/module/output_MaskRcnn_ORIGA/healthy/",
        )
        config = MRCNN.Config(
            IS_INFERENCE=False,
            USE_GPU=True,
            DEBUG=True,
            WIDTH=1024,
            NUM_CLASSES=2,
            MASK_COLOR=helpers.COLORS['red'],
            MASK_PATHS={
                "Disc": os.path.join(
                    DATA_DIR,
                    "A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/5. Optic Disc",
                ),
            },
            IMAGE_PATH=os.path.join(
                DATA_DIR, "A. Segmentation/1. Original Images/a. Training Set/"
            ),
            WEIGHTS_PATH=f"{HOME}/mask_rcnn_coco.h5",
            MODEL_DIR="/home/thomas/TeleOphtalmo/module/models/branch2/",
            LEARNING_RATE=0.0001,
            EPOCHS=N_EPOCHS,
            cropped_image=cropped_image_config,
        )
        model = MRCNN.Model(config)
        model.train()

        best_model_path = model.get_best_model_path()
        config.IS_INFERENCE = True
        config.WEIGHTS_PATH = best_model_path
        best_model = Model(config)

        return best_model

    def crop_images(self):
        self.train_mrcnn().create_cropped_image()

    def train(self):
        config = resnet50.Config(
            resnet50.FreezeConfig(
                FREEZE_TYPE=resnet50.freeze_type["FREEZE_TO"],
                FREEZE_TO=-2,
            ),
            TRAIN_DATA_PATH_ROOT="/home/thomas/TeleOphtalmo/module/output_MaskRcnn_ORIGA/",
            MODEL_DIR="/home/thomas/TeleOphtalmo/module/models/branch2",
            EPOCHS=N_EPOCHS,
        )
        model = resnet50.Model(config)
        model.train()
        return model

    def infer(self):
        config = resnet50.Config(
            INFERENCE_DATA_PATH_ROOT="/home/jupyter/Data/valid",
            IS_INFERENCE=True,
            MODEL_DIR="/home/thomas/TeleOphtalmo/module/models/branch2",
            MODEL_NAME="best_model.pkl",
        )
        model = resnet50.Model(config)
        return model.get_results()


class Branch3:
    def __init__(self):
        pass

    def get_train_config(self):
        DATA_DIR = f"{HOME}/Third_branch/mask_for_maskrcnn/"
        return MRCNN.Config(
            IS_INFERENCE=False,
            USE_GPU=True,
            DEBUG=True,
            WIDTH=1024,
            NUM_CLASSES=3,
            MASK_COLOR=helpers.COLORS['blue'],
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
            WEIGHTS_PATH=f"{HOME}/mask_rcnn_coco.h5",
            MODEL_DIR="/home/thomas/TeleOphtalmo/module/models/branch3/",
            LEARNING_RATE=0.0001,
            EPOCHS=N_EPOCHS,
        )

    def get_infer_config(self):
        config = self.get_train_config()
        config.IS_INFERENCE = True
        return config

    def train(self):
        config = self.get_train_config()
        model = MRCNN.Model(config)
        model.train()

        best_model_path = model.get_best_model_path()
        config.IS_INFERENCE = True
        config.WEIGHTS_PATH = best_model_path
        best_model = MRCNN.Model(config)

        return best_model

    def infer(self):
        config = self.get_inference_config()
        model = MRCNN.Model(config)
        return model.infer()


class Model:
    def __init__(self):
        self.branch1 = Branch1()
        self.branch2 = Branch2()
        self.branch3 = Branch3()

    def prepare_branch2_dataset(self):
        self.branch2.crop_images()
        helpers.train_valid_split(
            "/home/thomas/TeleOphtalmo/module/output_MaskRcnn_ORIGA",
            "healthy",
            "glaucoma",
        )

    def train(self):
        self.branch1.train()
        self.branch2.train()
        self.branch3.train()

    def infer(self):
        results_branch1 = self.branch1.infer()
        results_branch2 = self.branch2.infer()
        results_branch3 = self.branch3.infer()
        return [results_branch1, results_branch2, results_branch3]


if __name__ == "__main__":
    model = Model()

    # create branch 2 cropped dataset if needed
    model.prepare_branch2_dataset()

    # train all branches sequentially
    model.train()
