import unittest
from typing import List

import numpy as np
import pandas as pd
from skimage import io


# monkey patch cv2.imread before import helpers
def imread_mock(path):
    return np.array([
        [[1, 0, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0]],
        [[0, 1, 0], [0, 0, 1]],
    ])


io.imread = imread_mock


import helpers


class MRCNNModelMock:
    def __init__(self):
        pass

    def detect(self, images: List[str], verbose: int = 1):
        masks = np.array([
            [[1, 0, 0, 0, 0], [0, 1, 1, 0, 1]],
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
            [[1, 0, 0, 0, 0], [0, 1, 1, 0, 0]],
        ])
        res = {
            "rois": [i for i in range(5)],
            "class_ids": [1, 2, 1, 2, 2],
            "scores": [0.2, 0.7, 0.9, 0.1, 0.95],
            "masks": masks,
        }
        return [res for i in images]


class TestHelpers(unittest.TestCase):
    def test_filter_best_mrcnn_results(self):
        result = MRCNNModelMock().detect(["some_image"])[0]
        self.assertEqual(helpers.get_best_mrcnn_result_index_for_class(result, 1), 2)
        self.assertEqual(helpers.get_best_mrcnn_result_index_for_class(result, 2), 4)

    def test_mrcnn_iou_eval(self):
        model = MRCNNModelMock()
        n_mask_classes = 2
        annotations = pd.DataFrame(
            [
                ["Some_Path", 2, 3],
            ],
            columns=["Paths", "A", "B"],
        )
        col_names = ["A", "B"]
        result = helpers.mrcnn_iou_eval(model, annotations, n_mask_classes, col_names)
        self.assertEqual(result == [[1.0], [0.5]])


if __name__ == "__main__":
    unittest.main()
