import os
from pathlib import Path
import json
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


class Branch_dataset:
    def __init__(
        self,
        branch_dic,
        score_column,
        split_charac,
        intext,
        num_index,
        num_index2,
        len_test,
        test_for_y,
        index_y_test,
    ):
        """Creating a df from the dictionary output of each branch
        branch_dic: the name of the dictionary
        score_column: the name to give to the score column, has to be in a list
        split_charac: the character to split the image name by to get image ID number
        intext: True if the ID number is mixed with string characters
        num_index: The index of the ID number after split
        num_index2: If intext == True, where the ID number starts in the string. If intext is False put 0
        len_test: True is a picture can be classified as glaucoma if the image name has a different number of part compared to healthy
        test_for_y: What identify the image as glaucoma from its name
        index_y_test: If len_test == False, the index of the string to verify, if len_test == True put 0"""

        df = pd.DataFrame.from_dict(branch_dic, orient="index", columns=score_column)
        df.reset_index(level=0, inplace=True)
        if intext == True:
            df["Img_number"] = (
                df["index"]
                .str.split(pat=split_charac)
                .map(lambda x: int(x[num_index][num_index2:]))
            )
        else:
            df["Img_number"] = (
                df["index"].str.split(pat=split_charac).map(lambda x: int(x[num_index]))
            )
        if len_test == True:
            df["Y"] = np.where(
                df["index"].str.split(pat=split_charac).map(lambda x: len(x))
                == test_for_y,
                1,
                0,
            )
        else:
            df["Y"] = np.where(
                df["index"].str.split(pat=split_charac).map(lambda x: x[index_y_test])
                == test_for_y,
                1,
                0,
            )

        self.df = df

    def merging_branches(self, branch2, branch3):

        self.df = self.df.merge(
            branch2[["score2", "Img_number"]],
            how="left",
            on="Img_number",
            left_index=True,
        )
        self.df = self.df.merge(
            branch3[["score3", "Img_number"]],
            how="left",
            on="Img_number",
            left_index=True,
        )
        return self.df


@dataclass
class Config:
    PATH_1_TRAIN: str = ""
    PATH_1_VAL: str = ""
    PATH_2_TRAIN: str = ""
    PATH_2_VAL: str = ""
    PATH_3_TRAIN: str = ""
    PATH_3_VAL: str = ""
    N_BRANCHES: int = 3
    MODEL_PATH: str = ""
    IS_INFERENCE: bool = False


class Model:
    def __init__(self, config: Config):
        self.config = config

        if self.config.IS_INFERENCE:
            self.load_model()
        else:
            self.init_dataset()

    def init_dataset(self):
        self.load_all_json()
        branch1_train = Branch_dataset(
            self.branch1_dic, ["score1"], "_", True, 0, 2, True, 3, 0
        )
        branch1_val = Branch_dataset(
            self.branch1_valid_dic, ["score1"], "_", True, 0, 2, True, 3, 0
        )

        branch2_train = Branch_dataset(
            self.branch2_dic, ["score2"], "_", False, 3, 0, False, "glaucoma", 0
        )
        branch2_val = Branch_dataset(
            self.branch2_valid_dic, ["score2"], "_", False, 3, 0, False, "glaucoma", 0
        )

        branch3_train = Branch_dataset(
            self.branch3_dic, ["score3"], "_", True, 0, 2, True, 3, 0
        )
        branch3_val = Branch_dataset(
            self.branch3_valid_dic, ["score3"], "_", True, 0, 2, True, 3, 0
        )
        self.multibranch_df = branch1_train.merging_branches(
            branch2_train.df, branch3_train.df
        )
        self.multibranch_valid_df = branch1_val.merging_branches(
            branch2_val.df, branch3_val.df
        )

        # Because of na we will need 3 classifier
        # The one using the 3 branches will need to drop the na
        if self.config.N_BRANCHES == 3:
            self.multibranch_df = self.multibranch_df.dropna()
            self.multibranch_valid_df = self.multibranch_valid_df.dropna()

    def get_dataset_col_names(self):
        if self.config.N_BRANCHES == 3:
            return ["score1", "score2", "score3"]
        else:
            return ["score1", "score2"]

    def train(self):
        cols = self.get_dataset_col_names()

        X = self.multibranch_df[cols]
        y = self.multibranch_df["Y"]

        self.model = LogisticRegression()
        self.model.fit(X, y)

        self.export_model()

    def score(self):
        cols = self.get_dataset_col_names()

        X_val = self.multibranch_valid_df[cols]
        y_val = self.multibranch_valid_df["Y"]

        return self.logreg_3branches.score(X_val, y_val)

    def load_all_json(self):
        with open(self.config.PATH_1_TRAIN) as json_file:
            self.branch1_dic = json.load(json_file)

        with open(self.config.PATH_1_VAL) as json_file:
            self.branch1_valid_dic = json.load(json_file)

        with open(self.config.PATH_2_TRAIN) as json_file:
            self.branch2_dic = json.load(json_file)

        with open(self.config.PATH_2_VAL) as json_file:
            self.branch2_valid_dic = json.load(json_file)

        with open(self.config.PATH_3_TRAIN) as json_file:
            self.branch3_dic = json.load(json_file)

        with open(self.config.PATH_3_VAL) as json_file:
            self.branch3_valid_dic = json.load(json_file)

    def export_model(self):
        model_path = Path(self.config.MODEL_PATH)
        os.makedirs(model_path.parent, exist_ok=True)
        pickle.dump(self.model, open(self.config.MODEL_PATH, "wb"))

    def load_model(self):
        file = open(self.config.MODEL_PATH, 'rb')
        self.model = pickle.load(file, encoding="ASCII")
