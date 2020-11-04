from dataclasses import dataclass

import numpy as np
from fastai.vision import (
    ImageDataBunch, Path, get_transforms,
    imagenet_stats, cnn_learner, accuracy,
    models, ClassificationInterpretation,
    open_image
)
from .helpers import SaveBestModel, fmod


@dataclass
class Config:
    PATH_ROOT: str


class Model:
    def __init__(self, config: Config):
        self.config = config
        self.data = self.load_data()
        self.init_learner()

    def init_learner(self):
        self.learner = cnn_learner(
            self.data,  # must be a data loader instance (dls) like ImageDataBunch.from_folder()
            models.resnet50,  # pre-trained model chosen
            metrics=accuracy,
            callback_fns=SaveBestModel)
        self.interpretation = ClassificationInterpretation.from_learner(self.learner)

    def load_data(self):
        # We create a DataBunch object from the Data folder
        return ImageDataBunch.from_folder(
            Path(self.config.PATH_ROOT),
            train="train",
            valid="valid",
            # train='.',
            # valid_pct=0.2, # ratio split for train & test
            ds_tfms=get_transforms(  # Dataset transformations (augmentation),
                do_flip=False,       # generates copies of the images. Here,
                flip_vert=False,     # no transformation applied.
                max_rotate=0,
                p_affine=0,
            ),
            size=(256, 256),  # size of the images created
            num_workers=4,
            bs=16,
        ).normalize(imagenet_stats)

    def get_results(self):
        return fmod(self.learner, self.data.train_ds.items)

    def predict(self, img_path: str):
        return model.learner.predict(open_image(img_path))


if __name__ == "__main__":
    np.random.seed(42)

    config = Config('Data/')
    model = Model(config)

    # Fully frozen model
    model.learner.fit_one_cycle(50)
    model.interpretation.plot_confusion_matrix()
    model.learner.save('resnet50_unfrozen', return_path=True)

    # Fully unfrozen
    model.init_learner()  # reset learner
    model.learner.load("resnet50_2unfrozen")
    model.learner.unfreeze()
    model.learner.lr_find()
    model.learner.recorder.plot()
    model.learner.fit_one_cycle(50, max_lr=6e-05)
    model.interpretation.plot_confusion_matrix()
    model.learner.save('resnet50_fully_unfrozen', return_path=True)

    # Freezing the last two layers
    model.init_learner()
    model.learner.freeze_to(-2)
    model.learner.fit_one_cycle(50)
    model.interpretation.plot_confusion_matrix()

    # Get 0 to 1 outputs
    model.init_learner()
    model.learner.load("resnet50_fullunfrozen")
    pred_class, pred_idx, outputs = model.predict("Data/Healthy/Im0141_ORIGA.jpg")
    print(pred_class, pred_idx, outputs, sep="\n")

    result_dict = fmod(model.learner, model.data.train_ds.items)
