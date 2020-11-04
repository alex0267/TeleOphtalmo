from fastai.vision import Recorder, MetricsList, open_image
from typing import Any


class SaveBestModel(Recorder):
    def __init__(self, learn, name="best_model"):
        super().__init__(learn)
        self.name = name
        self.best_loss = None
        self.best_acc = None
        self.save_method = self.save_when_acc

    def save_when_acc(self, metrics):
        loss, acc = metrics[0], metrics[1]
        if self.best_acc is None or acc > self.best_acc:
            self.best_acc = acc
            self.best_loss = loss
            self.learn.save(f"{self.name}")
            print("Save the best accuracy {:.5f}".format(self.best_acc))
        elif acc == self.best_acc and loss < self.best_loss:
            self.best_loss = loss
            self.learn.save(f"{self.name}")
            print("Accuracy is eq, Save the lower loss {:.5f}".format(self.best_loss))

    def on_epoch_end(self, last_metrics=MetricsList, **kwargs: Any):
        self.save_method(last_metrics)


def fmod(learn, list_images_paths, glaucoma_idx=0):
    '''
    fmod stands for: F_astai M_odel O_utput D_ictionary

    This function takes as its arguments a fastai model and a list of images path.
    The function returns a dictionary with the images name as the key and the images
    Glaucoma score as the associated value
    '''
    result_dic = {}
    for path in list_images_paths:
        path = str(path)
        pred_class, pred_idx, outputs = learn.predict(open_image(path))
        img_name = path.split('.')[0].split('/')[-1]
        result_dic[img_name] = float(outputs[glaucoma_idx])

    return result_dic
