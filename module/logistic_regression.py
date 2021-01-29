import os
from pathlib import Path
import json
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from typing import List


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

        :param branch_dic: the name of the dictionary containing branch1 data
        :param score_column: the name to give to the score column, has to be
            in a list
        :param split_charac: the character to split the image name by to get
           image ID number
        :param intext: True if the ID number is mixed with string characters
        :param num_index: The index of the ID number after split
        :param num_index2: If intext == True, where the ID number starts in the
            string. If intext is False put 0
        :param len_test: True is a picture can be classified as glaucoma if the
            image name has a different number of part compared to healthy
        :param test_for_y: What identify the image as glaucoma from its name
        :param index_y_test: If len_test == False, the index of the string to
            verify, if len_test == True put 0"""

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
        """Merges score dataframes from branch 2 and branch 3"""
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
    """Configuration for the logistic regression model.

    :param PATH_1_TRAIN: path to the branch 1 training dictionnary.
    :param PATH_1_VAL: path to the branch 1 validation dictionnary.
    :param PATH_2_TRAIN: path to the branch 2 training dictionnary.
    :param PATH_2_VAL: path to the branch 2 validation dictionnary.
    :param PATH_3_TRAIN: path to the branch 3 training dictionnary.
    :param PATH_3_VAL: path to the branch 3 validation dictionnary.
    :param N_BRANCHES: use 2 or 3 branches in the prediciton.
    :param MODEL_PATH: path to the model to load if in inference mode or
        to save to if in train mode.
    :param IS_INFERENCE: is the model being used to infer or to train.
    """
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
        """Instanciate the logistic regression model.

        :param config: configuration to use."""
        self.config = config

        if self.config.IS_INFERENCE:
            self.load_model()
        else:
            self.init_dataset()

    def init_dataset(self):
        """Constructs the dataset using the output dictionary from:
            - the two first branches if `config.N_BRANCHES == 2`
            - the all three branches if `config.N_BRANCHES == 3`
        This method sets two attributes that allow easy acces to the train
        and validation datasets.
        """
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

    def get_dataset_col_names(self) -> List[str]:
        """Returns the column names from the model input to be used when training
        and scoring the model.

        :returns: a list of column names"""
        if self.config.N_BRANCHES == 3:
            return ["score1", "score2", "score3"]
        else:
            return ["score1", "score2"]

    def train(self):
        """Train the logistic regression."""
        cols = self.get_dataset_col_names()

        X = self.multibranch_df[cols]
        y = self.multibranch_df["Y"]

        self.model = LogisticRegression()
        self.model.fit(X, y)

        self.export_model()

    def score(self):
        """Score the logistic regression"""
        cols = self.get_dataset_col_names()

        X_val = self.multibranch_valid_df[cols]
        y_val = self.multibranch_valid_df["Y"]

        return self.model.score(X_val, y_val)

    def load_all_json(self):
        """Loads the train and validation output dictionaries from
        all previous branches into attributes."""
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
        """Export the model to a file."""
        model_path = Path(self.config.MODEL_PATH)
        os.makedirs(model_path.parent, exist_ok=True)
        pickle.dump(self.model, open(self.config.MODEL_PATH, "wb"))

    def load_model(self):
        """Load the model from a file."""
        file = open(self.config.MODEL_PATH, 'rb')
        self.model = pickle.load(file, encoding="ASCII")
