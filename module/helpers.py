import os
import re
from typing import Any

import cv2
import numpy as np
import pandas as pd
from fastai.vision import MetricsList, Recorder, open_image
from imgaug import augmenters as iaa
from mrcnn import utils
from skimage.io import imread


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
    """
    fmod stands for: F_astai M_odel O_utput D_ictionary

    This function takes as its arguments a fastai model and a list of images path.
    The function returns a dictionary with the images name as the key and the images
    Glaucoma score as the associated value
    """
    result_dic = {}
    for path in list_images_paths:
        path = str(path)
        pred_class, pred_idx, outputs = learn.predict(open_image(path))
        img_name = path.split(".")[0].split("/")[-1]
        result_dic[img_name] = float(outputs[glaucoma_idx])

    return result_dic


def resizeAndPad(img, size, padColor=0):
    """Resize the image keeping the height proportional to the width."""
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
    if len(img.shape) == 3 and not isinstance(
        padColor, (list, tuple, np.ndarray)
    ):  # color image but only one color provided
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


def imshow_components(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, labels = cv2.connectedComponents(gray)
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0
    return labeled_img


def mask_fct(img, WIDTH, LENGTH):
    """Two fonction for create binary mask for keras"""
    img = imshow_components(img)
    colors = np.unique(img.reshape(-1, img.shape[-1]), axis=0)
    colors = colors[1:]
    masks = None
    save_img = img.copy()
    for index, color in enumerate(colors):
        img = save_img.copy()
        mask = (img == color).all(axis=2)
        img[np.invert(mask)] = [0, 0, 0]
        img = img.astype(bool)
        mask = np.logical_or.reduce(img, axis=2)
        if index == 0:
            masks = mask.copy()
            masks = np.reshape(masks, (WIDTH, LENGTH, 1))
        else:
            masks = np.dstack((masks, mask))
    class_ids = np.zeros((len(colors)), dtype=np.int32)
    class_ids[:] = 1
    return (masks, class_ids)


def rename(row):
    if "flip" in row:
        return row[:-12] + "_flip.jpg"
    return row[:-7] + ".jpg"


def create_pathology_dataframe(mask_path, image_path):
    """create dataframe for pathology selected"""
    files = os.listdir(mask_path)
    files = files[0:1]
    images = []
    for path in files:
        path = os.path.join(mask_path, path)
        paths = os.listdir(path)
        for img in paths:
            img = os.path.join(path, img)
            images.append(img)
    annotations = pd.DataFrame(images, columns=["Paths_mask"])
    annotations["ID_mask"] = annotations["Paths_mask"].apply(
        lambda row: row.split(os.path.sep)[-1]
    )
    annotations["ID"] = annotations["ID_mask"].apply(lambda row: rename(row))
    annotations["Path"] = annotations["ID"].apply(lambda row: image_path + row)
    return annotations


class DetectorDataset(utils.Dataset):
    """Dataset class for training our dataset."""

    def __init__(self, image_fps, image_annotations, image_path, shape):
        super().__init__(self)

        self.shape = shape

        # Add classes
        self.add_class("point", 1, "point")
        # add images
        for i, fp in enumerate(image_fps):
            annotations = image_annotations.query('ID =="' + fp + '"')[
                "Paths_mask"
            ].iloc[0]
            self.add_image(
                "point",
                image_id=i,
                path=os.path.join(image_path, fp),
                annotations=annotations,
                orig_height=shape[0],
                orig_width=shape[0],
            )

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info["path"]

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info["path"]
        image = imread(fp)
        image = resizeAndPad(image, self.shape)
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask = cv2.imread(info["annotations"])
        mask = resizeAndPad(mask, self.shape)
        tab_mask = mask_fct(mask)
        return tab_mask


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


# path folder for testing images
def create_cropped_image(model, input_path, name_path, output_path, shape):
    # Image augmentation (light but constant)
    aug_detect = iaa.Sequential([iaa.CLAHE()])

    path = input_path
    images = sorted_alphanumeric(os.listdir(path))
    # print(images)

    for name in images:
        img = imread(path + name)
        img_detect = img.copy()
        print(name)
        aug = False
        if aug:
            img_detect = aug_detect(image=img)
        # make prediction
        results = model.detect([img_detect], verbose=1)
        # visualize the results
        r = results[0]
        if r["rois"].size > 0:  # if Roi is found.
            # visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], ['BG', 'point'], r['scores'],
            #                 )
            y, x, h, w = r["rois"][0]
            # cropped_image = tf.image.crop_to_bounding_box(img, y-30, x-30, y-(y-30) , x-(x-30))
            # print(cropped_image)
            # plt.imshow(cropped_image)
            # crop_img = img[y:y+h, x:x+w]
            # plt.imshow("cropped", crop_img)
            # cv2.imread(cropped_image)
            y, x = y - 30, x - 30
            h, w = h + 30, w + 30
            roi = img[y:h, x:w, :]
            roi_resized = resizeAndPad(roi, shape)
            # print(roi_resized.shape)
            # plt.imshow(roi_resized)
            # plt.imshow(roi)
            i = int(name.split("_")[0][2:6])
            cv2.imwrite(
                output_path + name_path + "_roi_resized_{0}.png".format(i),
                cv2.cvtColor(roi_resized, cv2.COLOR_RGB2BGR),
            )

        else:  # Roi not found
            print("roi not found")


def mrcnn_iou_eval(model, anns):
    """
    Evaluation of the roi and masks provided by the mrcnn model

    model: the model we want to evaluate
    anns: a dataframe with the filepaths of the evaluation images and masks.

    The output is:
    list_iou: a list of iou values
    """

    list_iou = []
    for idx in range(len(anns)):
        path = anns.loc[idx, "Path"]
        img = imread(path)
        img_detect = img.copy()
        results = model.detect([img_detect], verbose=1)

        mask_disc_org = cv2.imread(anns.loc[idx, "Paths_mask"])
        mask_disc_org = np.where(mask_disc_org > 0, 1, 0)

        target = mask_disc_org[:, :, 2]
        prediction = results[0]["masks"][:, :, 0]

        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        iou_score = np.sum(intersection) / np.sum(union)

        list_iou.append(iou_score)

    return list_iou