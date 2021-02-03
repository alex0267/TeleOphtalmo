import os
import re
import shutil
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from fastai.basic_train import Learner
from fastai.vision import MetricsList, Recorder, open_image
from imgaug import augmenters as iaa
from mrcnn import utils
from mrcnn.model import MaskRCNN
from skimage.io import imread

COLORS = {
    "blue": 0,
    "green": 1,
    "red": 2,
}


class SaveBestModel(Recorder):
    """Saves the best model based on scoring metric"""

    def __init__(self, learn: Learner, name: str = "best_model"):
        """Instanciates a `SaveBestModel` object.

        :param learn: learner to save the best model for.
        :param name: name of file to save."""
        super().__init__(learn)
        self.name = name
        self.best_loss = 0.0
        self.best_acc = 0.0
        self.save_method = self.save_when_acc

    def save_when_acc(self, metrics: Tuple[float, float]):
        """Callback used after each epoch to save the model if it is the best.

        :param metric: tuple containing the current loss as first element and
            the current accuracy as second element."""
        loss, acc = metrics[0], metrics[1]
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_loss = loss
            self.learn.export(f"{self.name}.pkl")
            print("Save the best accuracy {:.5f}".format(self.best_acc))
        elif acc == self.best_acc and loss < self.best_loss:
            self.best_loss = loss
            self.learn.export(f"{self.name}.pkl")
            print("Accuracy is eq, Save the lower loss {:.5f}".format(self.best_loss))

    def on_epoch_end(self, last_metrics=MetricsList, **kwargs: Any):
        self.save_method(last_metrics)


def fmod(
    learn: Learner, list_images_paths: List[str], glaucoma_idx: int = 0
) -> Dict[str, float]:
    """
    fmod stands for: F_astai M_odel O_utput D_ictionary.

    :param learn: fast ai learner.
    :param list_images_paths: list of image paths to include in the output
        dictionary.
    :param glaucoma_idx: glaucoma label.
    :return: a dictionary with the images name as the key and the images
        glaucoma score as the associated value.
    """
    result_dic = {}
    for path in list_images_paths:
        path = str(path)
        pred_class, pred_idx, outputs = learn.predict(open_image(path))
        img_name = path.split(".")[0].split("/")[-1]
        result_dic[img_name] = float(outputs[glaucoma_idx])

    return result_dic


def resizeAndPad(
    img: np.ndarray, size: Tuple[int, int], padColor: Union[int, Iterable[int]] = 0
) -> np.ndarray:
    """Resize the image keeping the height proportional to the width.

    :param img: image to resize and pad as a Numpy array.
    :param size: size of resized image.
    :param padColor: color used for padded pixels.
    :return: resized image."""
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA

    else:  # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = float(w) / h
    saspect = float(sw) / sh

    if (saspect > aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(
            int
        )
        pad_top, pad_bot = 0, 0

    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    # set pad color
    if len(img.shape) == 3 and isinstance(padColor, int):
        # color image but only one color provided
        padColor = [padColor] * 3

    # scale and padBORDER_CONSTANT, value=padColor)

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(
        scaled_img,
        pad_top,
        pad_bot,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=padColor,
    )

    return scaled_img


def create_pathology_dataframe(
    image_path: str, mask_paths: Dict[str, str]
) -> pd.DataFrame:
    """Creates a dataframe mapping image paths to masks.

    :param image_path: path to the directory containing the images
        to include in the dataframe.
    :param mask_paths: a dictionnary mapping mask names to directories
        containing the mask files.
    :return: a dataframe mapping image paths to their masks."""
    files_image = os.listdir(image_path)
    images = []
    for path in files_image:
        if path != ".ipynb_checkpoints":  # temp fix to a bug
            path = os.path.join(image_path, path)
            images.append(path)
            images.sort()

    annotations = pd.DataFrame(images, columns=["Paths"])
    annotations["ID"] = annotations["Paths"].apply(
        lambda row: row.split(os.path.sep)[-1]
    )

    for mask_name, mask_path in mask_paths.items():
        files = os.listdir(mask_path)
        masks = []
        for path in files:
            if path != ".ipynb_checkpoints":  # temp fix to a bug
                path = os.path.join(mask_path, path)
                masks.append(path)
                masks.sort()

        annotations[mask_name] = masks

    return annotations


class DetectorDataset(utils.Dataset):
    """Dataset class for training our dataset."""

    def __init__(
        self,
        image_fps: List[str],
        image_annotations: pd.DataFrame,
        shape: Tuple[int, int],
        class_names: List[str],
        annotation_mask_names: List[str],
        mask_color: str,
    ):
        """Instanciate a detector dataset object.

        :param image_fps: list of image file paths to include in the dataset.
        :param image_annotations: dataframe mapping image paths to masks.
        :param shape: shape of images.
        :param class_names: list of all class names in the dataset.
        :param annotation_mask_names: list of all mask names in the dataset.
        :param mask_color: color of the mask in the mask images."""
        super().__init__(self)

        self.shape = shape
        self.annotation_mask_names = annotation_mask_names
        self.mask_color = mask_color

        # Add classes
        for i, class_name in enumerate(class_names):
            self.add_class("Magrabia", i, class_name)

        # add images
        for i, fp in enumerate(image_fps):
            path_img = image_annotations.query('ID =="' + fp + '"')["Paths"].iloc[0]
            annotations = []
            for mask_name in annotation_mask_names:
                annotations.append(
                    image_annotations.query('ID =="' + fp + '"')[mask_name].iloc[0]
                )
            self.add_image(
                "Magrabia",
                image_id=i,
                path=path_img,
                annotations=annotations,
                orig_width=shape[0],
                orig_height=shape[1],
            )

    def image_reference(self, image_id) -> str:
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        :param image_id: id of image to return a reference for.
        :return: the path to `image_id`."""
        info = self.image_info[image_id]
        return info["path"]

    def load_image(self, image_id) -> np.ndarray:
        """Load the specified image.

        :param image_id: id of the image to load.
        :return: a [H,W,3] Numpy array."""
        info = self.image_info[image_id]
        fp = info["path"]
        image = imread(fp)
        image = resizeAndPad(image, self.shape)
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id) -> Tuple[List[bool], List[str]]:
        """Load instance masks for the given image.

        :param image_id: id of the image to load the masks for.
        :return: a tuple with:
            - a bool array of shape [height, width, instance count] with
              a binary mask per instance as first element
            - a 1D array of class IDs of the instance masks as second element.
        """
        n_classes = len(self.annotation_mask_names)
        masks = np.zeros((self.shape[0], self.shape[1], n_classes))
        for i in range(0, n_classes):
            info = self.image_info[image_id]
            mask = cv2.imread(info["annotations"][i])
            mask = resizeAndPad(mask, self.shape)
            mask = np.where(mask > 0, 1, 0)
            masks[:, :, i] = mask[:, :, self.mask_color]

        class_ids = np.array([i + 1 for i in list(range(n_classes))])

        return (masks, class_ids)


def sorted_alphanumeric(data: List[str]) -> List[str]:
    """Sort a list of alphanumeric strings.

    :param data: a list of alphanumeric strings.
    :return: `data` sorted."""

    def convert(text: str):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key: str):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(data, key=alphanum_key)


# path folder for testing images
def create_cropped_image(
    model: MaskRCNN,
    input_path: str,
    name_path: str,
    output_path: str,
    shape: Tuple[int, int],
) -> List[str]:
    """Crops an input images using an MRCNN model.

    :param model: the MRCNN model to be used when cropping
    :param input_path: path to the folder containing the images to crop
    :param name_path: base name for output file
    :param output_path: path to the cropped image folder
    :param shape: shape of the cropped image

    :return: List of paths to the cropped images"""
    # Image augmentation (light but constant)
    aug_detect = iaa.Sequential([iaa.CLAHE()])

    path = input_path
    images = sorted_alphanumeric(os.listdir(path))
    # print(images)

    skipped_files = []
    cropped_image_paths = []
    for name in images:
        print("===> ", name)

        try:
            img = imread(os.path.join(path, name))
        except ValueError:
            skipped_files.append(os.path.join(path, name))
            continue

        img_detect = img.copy()
        aug = False
        if aug:
            img_detect = aug_detect(image=img)
        # make prediction
        results = model.detect([img_detect], verbose=1)
        # visualize the results
        r = results[0]
        if r["rois"].size > 0:  # if Roi is found.
            y, x, h, w = r["rois"][0]
            y, x = y - 30, x - 30
            h, w = h + 30, w + 30
            roi = img[y:h, x:w, :]
            roi_resized = resizeAndPad(roi, shape)
            i = int(name.split("_")[0][2:6])
            target_path = os.path.join(
                output_path, name_path + "_roi_resized_{0}.png".format(i)
            )
            cropped_image_paths.append(target_path)
            cv2.imwrite(
                target_path,
                cv2.cvtColor(roi_resized, cv2.COLOR_RGB2BGR),
            )

        else:  # Roi not found
            print("roi not found")

    if len(skipped_files):
        print("The following files were skipped:")
        for path in skipped_files:
            print(path)

    return cropped_image_paths


MrcnnResultEntry = Dict[str, Any]


def get_best_mrcnn_result_index_for_class(
    mrcnn_result_entry: MrcnnResultEntry, class_id: int
) -> Optional[int]:
    """Given an mrcnn result entry.

    :return: the index of the best score for given class if 1 or more masks
        were detected and `None` otherwise."""
    best_index = None
    best_score = 0
    scores = mrcnn_result_entry.get("scores")
    if scores is not None:
        for i, score in enumerate(scores):
            class_ids = mrcnn_result_entry.get("class_ids")
            if class_ids is not None:
                if class_ids[i] == class_id:
                    if score > best_score:
                        best_score = score
                        best_index = i
    return best_index


def mrcnn_iou_eval(
    model: MaskRCNN, anns: pd.DataFrame, n_masks: int, col_names: List[str]
) -> List[List[float]]:
    """
    Evaluation of the roi and masks provided by the mrcnn model

    :param model: the model we want to evaluate
    :param anns: a dataframe with the filepaths of the evaluation images and
        masks.
    :param n_masks: number of masks
    :param col_names: name of the mask annotation columns

    :return: a list of iou values
    """
    spec: List[Tuple[List[float], str]] = [([], col_names[i]) for i in range(n_masks)]
    for idx in range(len(anns)):
        path = anns.loc[idx, "Paths"]
        img = imread(path)
        img_detect = img.copy()
        results = model.detect([img_detect], verbose=1)
        result = results[0]

        for i, spec_entry in enumerate(spec):
            class_id = i + 1
            list_iou, col_name = spec_entry
            class_ids = result.get("class_ids")
            if class_id in class_ids:
                best_class_score_index = get_best_mrcnn_result_index_for_class(
                    result, class_id
                )
                mask_org = imread(anns.loc[idx, col_name])
                mask_org = np.where(mask_org > 0, 1, 0)

                target = mask_org[:, :, 2]
                prediction = result.get("masks")[:, :, best_class_score_index]

                intersection = np.logical_and(target, prediction)
                union = np.logical_or(target, prediction)
                iou_score = np.sum(intersection) / np.sum(union)

                list_iou.append(iou_score)
            else:
                list_iou.append(0)

    return [spec_entry[0] for spec_entry in spec]


def train_valid_split(data_dir: str, healthy_name: str, glaucoma_name: str):
    """Split """
    # The list of Origa images in the train files
    # TODO remove hard coded file paths
    list_healthy_train_1 = os.listdir("/app/datasets/ORIGA/train/Healthy")
    list_glaucoma_train_1 = os.listdir("/app/datasets/ORIGA/train/Glaucoma")

    # The pictures ID of the images in the train files
    list_train_healthy = [int(pth.split("_")[0][3:]) for pth in list_healthy_train_1]
    list_train_glaucoma = [int(pth.split("_")[0][3:]) for pth in list_glaucoma_train_1]

    # All images resized for the second branch
    list_healthy_2 = os.listdir(os.path.join(data_dir, healthy_name))
    list_glaucoma_2 = os.listdir(os.path.join(data_dir, glaucoma_name))

    # Putting in the valid folder only the images that are in the valid folder of the 1st branch
    path_healthy = os.path.join(data_dir, healthy_name)
    path_healthy_train = os.path.join(data_dir, "train", healthy_name)
    path_healthy_valid = os.path.join(data_dir, "valid", healthy_name)

    for path in [
        path_healthy_train,
        path_healthy_valid,
    ]:
        os.makedirs(path, exist_ok=True)

    for im in list_healthy_2:
        im_index = int(im.split("_")[3][:-4])
        if im_index in list_train_healthy:
            shutil.copyfile(
                os.path.join(path_healthy, im), os.path.join(path_healthy_train, im)
            )
        else:
            shutil.copyfile(
                os.path.join(path_healthy, im), os.path.join(path_healthy_valid, im)
            )

    path_glaucoma = os.path.join(data_dir, glaucoma_name)
    path_glaucoma_train = os.path.join(data_dir, "train", glaucoma_name)
    path_glaucoma_valid = os.path.join(data_dir, "valid", glaucoma_name)

    for path in [
        path_glaucoma_train,
        path_glaucoma_valid,
    ]:
        os.makedirs(path, exist_ok=True)

    for im in list_glaucoma_2:
        im_index = int(im.split("_")[3][:-4])
        if im_index in list_train_glaucoma:
            shutil.copyfile(
                os.path.join(path_glaucoma, im), os.path.join(path_glaucoma_train, im)
            )
        else:
            shutil.copyfile(
                os.path.join(path_glaucoma, im), os.path.join(path_glaucoma_valid, im)
            )


def cup_to_disc_ratio(
    model: MaskRCNN, file_path: str
) -> Tuple[bool, Union[float, List[int]]]:
    """Calculates the cup/disc ratio for `file_path` using `model`.

    :param model: MaskRCNN model to use to calculate the masks.
    :param file_path: path to file to calculate ratio for.
    :return: a tuple with a boolean as first element set to `True` if the
        ratio could be calculated and `False` otherwise. The tuple's second
        element is the ratio if it could be calculated or a list of class ids
        that were found by the model if the ratio could not be calculated.
        this allows to see which mask could not be found (disc, cup or both)
        and prevented the ratio calculation."""
    img = imread(file_path)
    img_detect = img.copy()
    results = model.detect([img_detect], verbose=1)
    r = results[0]

    # Checking if both disc and cup where found
    if len(np.unique(r.get("class_ids"))) == 2:
        best_disc_index = get_best_mrcnn_result_index_for_class(r, 1)
        best_cup_index = get_best_mrcnn_result_index_for_class(r, 2)
        disc_pixel_sum = r.get("masks")[:, :, best_disc_index].sum()
        cup_pixel_sum = r.get("masks")[:, :, best_cup_index].sum()
        return True, cup_pixel_sum / disc_pixel_sum
    else:
        return False, r.get("class_ids")


def mmod(model: MaskRCNN, filenames: List[str]) -> Tuple[Dict[str, float], List[str]]:
    """fmod stands for: M_askRcnn M_odel O_utput D_ictionary

    :param model: model to be used to calculate the masks.
    :param filenames: list of path to images to calculate masks for.
    :return: a tuple with the first element being a dictionary with the images
        name as the key and the ratio of cup to disc as the associated value
        The second element being a list of images name the cup/disc ratio could
        not be calculated for."""
    result_dic: Dict[str, float] = {}
    list_failed_images = []
    for path in filenames:
        result = cup_to_disc_ratio(model, path)

        # Checking if both disc and cup where found
        if isinstance(result[1], float):
            ratio = result[1]
            img_name = path.split(".")[0].split("/")[-1]
            result_dic[img_name] = ratio
        else:
            class_ids = result[1]
            list_failed_images.append(img_name)
            print(
                "For picture {0} the model did not found a disc and a cup. class_ids: {1}".format(
                    path, class_ids
                )
            )

    return result_dic, list_failed_images


def mrcnn_b3_eval(
    model, anns, disc_annotation_colname, cup_annotation_colname
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Evaluation of the mask provided by the mrcnn model
    by comparing the square error of the cup/disc ratio
    of the annotated images compared to the infered masks.

    :param model: the model we want to evaluate
    :param anns: a dataframe with the filepaths of the evaluation images and
        masks.
    :param disc_annotation_colname: column name in `anns` that contains the
        disc masks.
    :param cup_annotation_colname: column name in `anns` that contains the
        cup masks.

    :return: a tuple with:
        - first element: df: a dataframe with the error between the ratio
        - second element: list_failed_images: a list of images where the model
            couldn't find a disc and a cup
    """
    df = pd.DataFrame(columns=["ID", "ann", "inf", "err"])
    list_failed_images = []
    for idx in range(len(anns)):
        path = anns.loc[idx, "Paths"]
        temp_dic = {}
        img = imread(path)
        img_detect = img.copy()
        results = model.detect([img_detect], verbose=1)
        r = results[0]

        # Checking if both disc and cup where found
        if len(np.unique(r.get("class_ids"))) == 2:
            best_disc_index = get_best_mrcnn_result_index_for_class(r, 1)
            best_cup_index = get_best_mrcnn_result_index_for_class(r, 2)
            disc_pixel_sum = r.get("masks")[:, :, best_disc_index].sum()
            cup_pixel_sum = r.get("masks")[:, :, best_cup_index].sum()
            ratio = cup_pixel_sum / disc_pixel_sum
            temp_dic["ID"] = anns.loc[idx, "ID"]
            temp_dic["inf"] = ratio

            mask_disc_org = imread(anns.loc[idx, disc_annotation_colname])
            mask_disc_org = np.where(mask_disc_org > 0, 1, 0)
            disc_pixel_sum_org = mask_disc_org[:, :, 0].sum()

            mask_cup_org = imread(anns.loc[idx, cup_annotation_colname])
            mask_cup_org = np.where(mask_cup_org > 0, 1, 0)
            cup_pixel_sum_org = mask_cup_org[:, :, 0].sum()

            ratio_org = cup_pixel_sum_org / disc_pixel_sum_org

            temp_dic["ann"] = ratio_org

            square_error = (ratio - ratio_org) * (ratio - ratio_org)
            absolute_error = square_error ** (1 / 2)
            temp_dic["err"] = absolute_error

            df = df.append(temp_dic, ignore_index=True)

        else:
            list_failed_images.append(anns.loc[idx, "ID"])
            print(
                "For picture {0} the model did not found a disc and a cup. class_ids: {1}".format(
                    path, r.get("class_ids")
                )
            )

    return df, list_failed_images
