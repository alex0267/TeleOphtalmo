import os

import helpers
import logistic_regression
import MRCNN
import resnet50

HOME = "/home/jupyter"
N_EPOCHS = 2


class Branch1:
    def __init__(self):
        pass

    def get_train_config(self):
        return resnet50.Config(
            resnet50.FreezeConfig(
                FREEZE_TYPE=resnet50.freeze_type["FREEZE_TO"],
                FREEZE_TO=-2,
            ),
            TRAIN_DATA_PATH_ROOT=f"{HOME}/Data/",
            MODEL_DIR="/home/thomas/TeleOphtalmo/module/models/branch1",
            EPOCHS=N_EPOCHS,
        )

    def get_infer_config(self):
        config = self.get_train_config()
        config.IS_INFERENCE = True
        config.INFERENCE_DATA_PATH_ROOT = "/home/jupyter/Data/valid/Glaucoma"
        config.MODEL_NAME = "best_model.pkl"
        return config

    def train(self):
        config = self.get_train_config()
        model = resnet50.Model(config)
        model.train()
        return model

    def infer(self):
        config = self.get_infer_config()
        export_path = "/home/thomas/TeleOphtalmo/module/output/b1"
        return resnet50.Model(config).export_dataset_output_dictionary(export_path)


class Branch2:
    def __init__(self):
        pass

    def get_mrcnn_train_config(self):
        DATA_DIR = f"{HOME}/Second_branch/data_train_mrcnn/"
        cropped_image_config = MRCNN.CroppedImageConfig(
            INPUT_PATH_GLAUCOMA="/home/jupyter/Data/Glaucoma/",
            INPUT_PATH_HEALTHY="/home/jupyter/Data/Healthy/",
            NAME_GLAUCOMA="glaucoma",
            NAME_HEALTHY="healthy",
            OUTPUT_PATH_GLAUCOMA="/home/thomas/TeleOphtalmo/module/output_MaskRcnn_ORIGA/glaucoma/",
            OUTPUT_PATH_HEALTHY="/home/thomas/TeleOphtalmo/module/output_MaskRcnn_ORIGA/healthy/",
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
            WEIGHTS_PATH=f"{HOME}/mask_rcnn_coco.h5",
            MODEL_DIR="/home/thomas/TeleOphtalmo/module/models/branch2/",
            LEARNING_RATE=0.0001,
            EPOCHS=N_EPOCHS,
            cropped_image=cropped_image_config,
        )

    def get_train_config(self):
        return resnet50.Config(
            resnet50.FreezeConfig(
                FREEZE_TYPE=resnet50.freeze_type["FREEZE_TO"],
                FREEZE_TO=-2,
            ),
            TRAIN_DATA_PATH_ROOT="/home/thomas/TeleOphtalmo/module/output_MaskRcnn_ORIGA/",
            MODEL_DIR="/home/thomas/TeleOphtalmo/module/models/branch2",
            EPOCHS=N_EPOCHS,
        )

    def get_infer_config(self):
        config = self.get_train_config()
        config.INFERENCE_DATA_PATH_ROOT = "/home/jupyter/Data/valid"
        config.IS_INFERENCE = True
        config.MODEL_NAME = "best_model.pkl"
        return config

    def train_mrcnn(self):
        config = self.get_mrcnn_train_config()
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
        config = self.get_train_config()
        model = resnet50.Model(config)
        model.train()
        return model

    def infer(self):
        config = self.get_infer_config()
        export_path = "/home/thomas/TeleOphtalmo/module/output/b2"
        return resnet50.Model(config).export_dataset_output_dictionary(export_path)


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
            WEIGHTS_PATH=f"{HOME}/mask_rcnn_coco.h5",
            MODEL_DIR="/home/thomas/TeleOphtalmo/module/models/branch3/",
            LEARNING_RATE=0.0001,
            EPOCHS=N_EPOCHS,
        )

    def get_infer_config(self):
        config = self.get_train_config()
        config.IS_INFERENCE = True
        config.WEIGHTS_PATH = (
            "/home/thomas/TeleOphtalmo/module/models/branch3/best_model.h5"
        )
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
        config = self.get_infer_config()
        model = MRCNN.Model(config)
        export_path = "/home/thomas/TeleOphtalmo/module/output/b3"
        return model.export_dataset_output_dictionary(export_path)


class LogReg:
    def __init__(self):
        pass

    def get_train_config(self):
        HOME = "/home/thomas/TeleOphtalmo/module/"
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

    def get_infer_config(self):
        config = self.get_train_config()
        config.IS_INFERENCE = True
        return config

    def train(self):
        model = logistic_regression.Model(self.get_train_config())
        model.train()

    def infer(self, X):
        model = logistic_regression.Model(self.get_infer_config())
        return model.perdict(X)


class Model:
    def __init__(self):
        self.branch1 = Branch1()
        self.branch2 = Branch2()
        self.branch3 = Branch3()
        self.logreg = LogReg()

    def prepare_branch2_dataset(self):
        self.branch2.crop_images()
        helpers.train_valid_split(
            "/home/thomas/TeleOphtalmo/module/output_MaskRcnn_ORIGA",
            "healthy",
            "glaucoma",
        )

    def train(self):
        HOME = "/home/thomas/TeleOphtalmo/module/"

        b1 = self.branch1.train()
        b1.export_dataset_output_dictionary(HOME)
        b2 = self.branch2.train()
        b2.export_dataset_output_dictionary(HOME)
        b3 = self.branch3.train()
        b3.export_dataset_output_dictionary(HOME)
        self.logreg.train()

    def infer(self):
        results_branch1 = self.branch1.infer()
        results_branch2 = self.branch2.infer()
        results_branch3 = self.branch3.infer()
        X = [[results_branch1, results_branch2, results_branch3]]
        # TODO handle 2 vs 3 branches logreg inference
        return self.logreg.infer(X)


if __name__ == "__main__":
    model = Model()

    # create branch 2 cropped dataset if needed
    model.prepare_branch2_dataset()

    # train all branches sequentially
    model.train()
