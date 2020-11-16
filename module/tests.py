import unittest
from typing import List

import numpy as np
import pandas as pd
from skimage import io


# monkey patch cv2.imread before import helpers
def imread_mock(path):
    return np.array(
        [
            [[1, 0, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 0, 1]],
        ]
    )


io.imread = imread_mock


import helpers

mrcnn_mock_mode = {
    "DISC_AND_CUP": 0,
    "DISC_ONLY": 1,
    "CUP_ONLY": 2,
    "NOTHING": 3,
}


class MRCNNModelMock:
    def __init__(self, mode):
        self.mode = mode

    def detect(self, images: List[str], verbose: int = 1):
        masks = np.array(
            [
                [[1, 0, 0, 0, 0], [0, 1, 1, 0, 1]],
                [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
                [[1, 0, 0, 0, 0], [0, 1, 1, 0, 0]],
            ]
        )
        res = {
            "rois": [i for i in range(5)],
            "scores": [0.2, 0.7, 0.9, 0.1, 0.95],
            "masks": masks,
        }
        if self.mode == mrcnn_mock_mode["DISC_AND_CUP"]:
            res["class_ids"] = [1, 2, 1, 2, 2]
        elif self.mode == mrcnn_mock_mode["DISC_ONLY"]:
            res["class_ids"] = [1, 1, 1, 1, 1]
        elif self.mode == mrcnn_mock_mode["CUP_ONLY"]:
            res["class_ids"] = [2, 2, 2, 2, 2]
        elif self.mode == mrcnn_mock_mode["NOTHING"]:
            res = {
                "rois": [],
                "class_ids": [],
                "scores": [],
                "masks": np.array([[[]]]),
            }
        return [res for i in images]


class TestHelpers(unittest.TestCase):
    def test_mrcnn_b3_eval(self):
        annotations = pd.DataFrame(
            [
                ["Some_Path", 2, 3, 1],
            ],
            columns=["Paths", "A", "B", "ID"],
        )

        model = MRCNNModelMock(mrcnn_mock_mode["DISC_AND_CUP"])
        filenames = ["Some_Path"]
        result_df, list_failed_images = helpers.mrcnn_b3_eval(
            model, annotations, "A", "B"
        )
        self.assertEqual(result_df[result_df["ID" == 1]]["inf"], 1 / 2)
        self.assertEqual(result_df[result_df["ID" == 1]]["ann"], 2 / 2)
        self.assertEqual(result_df[result_df["ID" == 1]]["err"], 0.5)

        model = MRCNNModelMock(mrcnn_mock_mode["DISC_ONLY"])
        result_df, list_failed_images = helpers.mrcnn_b3_eval(
            model, annotations, "A", "B"
        )
        self.assertEqual(list_failed_images, filenames)
        self.assertEqual(len(result_df), 0)

        model = MRCNNModelMock(mrcnn_mock_mode["CUP_ONLY"])
        result_df, list_failed_images = helpers.mrcnn_b3_eval(
            model, annotations, "A", "B"
        )
        self.assertEqual(list_failed_images, filenames)
        self.assertEqual(len(result_df), 0)

        model = MRCNNModelMock(mrcnn_mock_mode["NOTHING"])
        result_df, list_failed_images = helpers.mrcnn_b3_eval(
            model, annotations, "A", "B"
        )
        self.assertEqual(list_failed_images, filenames)
        self.assertEqual(len(result_df), 0)

    def test_mmod(self):
        model = MRCNNModelMock(mrcnn_mock_mode["DISC_AND_CUP"])
        filenames = ["Some_Path"]
        result_dict, list_failed_images = helpers.mmod(model, filenames)
        self.assertEqual(result_dict[filenames[0]], 1 / 2)

        model = MRCNNModelMock(mrcnn_mock_mode["DISC_ONLY"])
        result_dict, list_failed_images = helpers.mmod(model, filenames)
        self.assertEqual(list_failed_images, filenames)
        self.assertEqual(len(result_dict), 0)

        model = MRCNNModelMock(mrcnn_mock_mode["CUP_ONLY"])
        result_dict, list_failed_images = helpers.mmod(model, filenames)
        self.assertEqual(list_failed_images, filenames)
        self.assertEqual(len(result_dict), 0)

        model = MRCNNModelMock(mrcnn_mock_mode["NOTHING"])
        result_dict, list_failed_images = helpers.mmod(model, filenames)
        self.assertEqual(list_failed_images, filenames)
        self.assertEqual(len(result_dict), 0)

    def test_filter_best_mrcnn_results(self):
        result = MRCNNModelMock(mrcnn_mock_mode["DISC_AND_CUP"]).detect(["some_image"])[
            0
        ]
        self.assertEqual(helpers.get_best_mrcnn_result_index_for_class(result, 1), 2)
        self.assertEqual(helpers.get_best_mrcnn_result_index_for_class(result, 2), 4)

    def test_mrcnn_iou_eval(self):
        n_mask_classes = 2
        annotations = pd.DataFrame(
            [
                ["Some_Path", 2, 3],
            ],
            columns=["Paths", "A", "B"],
        )
        col_names = ["A", "B"]

        model = MRCNNModelMock(mrcnn_mock_mode["DISC_AND_CUP"])
        result = helpers.mrcnn_iou_eval(model, annotations, n_mask_classes, col_names)
        self.assertEqual(result, [[1.0], [0.5]])

        model = MRCNNModelMock(mrcnn_mock_mode["DISC_ONLY"])
        result = helpers.mrcnn_iou_eval(model, annotations, n_mask_classes, col_names)
        self.assertEqual(result, [[0.5], [0]])

        model = MRCNNModelMock(mrcnn_mock_mode["CUP_ONLY"])
        result = helpers.mrcnn_iou_eval(model, annotations, n_mask_classes, col_names)
        self.assertEqual(result, [[0], [0.5]])

        model = MRCNNModelMock(mrcnn_mock_mode["NOTHING"])
        result = helpers.mrcnn_iou_eval(model, annotations, n_mask_classes, col_names)
        self.assertEqual(result, [[0], [0]])


if __name__ == "__main__":
    unittest.main()
